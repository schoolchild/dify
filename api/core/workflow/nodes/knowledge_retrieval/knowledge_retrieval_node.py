import json
import logging
import re
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Optional, cast

from sqlalchemy import Float, and_, func, or_, text
from sqlalchemy import cast as sqlalchemy_cast
from sqlalchemy.orm import Session

from core.app.app_config.entities import DatasetRetrieveConfigEntity
from core.app.entities.app_invoke_entities import ModelConfigWithCredentialsEntity
from core.entities.agent_entities import PlanningStrategy
from core.entities.model_entities import ModelStatus
from core.model_manager import ModelInstance, ModelManager
from core.model_runtime.entities.message_entities import PromptMessageRole
from core.model_runtime.entities.model_entities import ModelFeature, ModelType
from core.model_runtime.model_providers.__base.large_language_model import LargeLanguageModel
from core.prompt.simple_prompt_transform import ModelMode
from core.rag.datasource.retrieval_service import RetrievalService
from core.rag.entities.metadata_entities import Condition, MetadataCondition
from core.rag.retrieval.dataset_retrieval import DatasetRetrieval
from core.rag.retrieval.retrieval_methods import RetrievalMethod
from core.variables import StringSegment
from core.variables.segments import ArrayObjectSegment
from core.workflow.entities.node_entities import NodeRunResult
from core.workflow.entities.workflow_node_execution import WorkflowNodeExecutionStatus
from core.workflow.nodes.enums import NodeType
from core.workflow.nodes.event.event import ModelInvokeCompletedEvent
from core.workflow.nodes.knowledge_retrieval.template_prompts import (
    METADATA_FILTER_ASSISTANT_PROMPT_1,
    METADATA_FILTER_ASSISTANT_PROMPT_2,
    METADATA_FILTER_COMPLETION_PROMPT,
    METADATA_FILTER_SYSTEM_PROMPT,
    METADATA_FILTER_USER_PROMPT_1,
    METADATA_FILTER_USER_PROMPT_2,
    METADATA_FILTER_USER_PROMPT_3,
)
from core.workflow.nodes.llm.entities import LLMNodeChatModelMessage, LLMNodeCompletionModelPromptTemplate
from core.workflow.nodes.llm.node import LLMNode
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from libs.json_in_md_parser import parse_and_check_json_markdown
from models.dataset import Dataset, DatasetMetadata, Document, RateLimitLog
from services.feature_service import FeatureService

from .entities import KnowledgeRetrievalNodeData, ModelConfig
from .exc import (
    InvalidModelTypeError,
    KnowledgeRetrievalNodeError,
    ModelCredentialsNotInitializedError,
    ModelNotExistError,
    ModelNotSupportedError,
    ModelQuotaExceededError,
)

logger = logging.getLogger(__name__)

default_retrieval_model = {
    "search_method": RetrievalMethod.SEMANTIC_SEARCH.value,
    "reranking_enable": False,
    "reranking_model": {"reranking_provider_name": "", "reranking_model_name": ""},
    "top_k": 2,
    "score_threshold_enabled": False,
}


class KnowledgeRetrievalNode(LLMNode):
    _node_data_cls = KnowledgeRetrievalNodeData  # type: ignore
    _node_type = NodeType.KNOWLEDGE_RETRIEVAL

    @classmethod
    def version(cls):
        return "1"

    def _run(self) -> NodeRunResult:  # type: ignore
        node_data = cast(KnowledgeRetrievalNodeData, self.node_data)
        # extract variables
        variable = self.graph_runtime_state.variable_pool.get(node_data.query_variable_selector)
        if not isinstance(variable, StringSegment):
            return NodeRunResult(
                status=WorkflowNodeExecutionStatus.FAILED,
                inputs={},
                error="Query variable is not string type.",
            )
        query = variable.value
        variables = {"query": query}
        
        # extract dataset_ids from variable if provided # 如果变量中提供了dataset_ids，则从中提取出来
        dataset_ids = node_data.dataset_ids
        if node_data.dataset_ids_variable_selector:
            print("1.dataset_ids_variable_selector: ", node_data.dataset_ids_variable_selector)
            dataset_ids_variable = self.graph_runtime_state.variable_pool.get(node_data.dataset_ids_variable_selector)
            if dataset_ids_variable:
                if isinstance(dataset_ids_variable, StringSegment):
                    # parse comma-separated string of dataset IDs # 解析以逗号分隔的数据集ID字符串
                    dataset_ids = [id.strip() for id in dataset_ids_variable.value.split(",") if id.strip()]
                elif isinstance(dataset_ids_variable, ArrayObjectSegment):
                    # extract dataset IDs from array # 从数组中提取数据集ID
                    dataset_ids = [str(item) for item in dataset_ids_variable.value if item]
                variables["dataset_ids"] = dataset_ids
                print("2.dataset_ids: ", dataset_ids)

        if not query:
            return NodeRunResult(
                status=WorkflowNodeExecutionStatus.FAILED, inputs=variables, error="Query is required."
            )
        # TODO(-LAN-): Move this check outside.
        # check rate limit
        knowledge_rate_limit = FeatureService.get_knowledge_rate_limit(self.tenant_id)
        if knowledge_rate_limit.enabled:
            current_time = int(time.time() * 1000)
            key = f"rate_limit_{self.tenant_id}"
            redis_client.zadd(key, {current_time: current_time})
            redis_client.zremrangebyscore(key, 0, current_time - 60000)
            request_count = redis_client.zcard(key)
            if request_count > knowledge_rate_limit.limit:
                with Session(db.engine) as session:
                    # add ratelimit record
                    rate_limit_log = RateLimitLog(
                        tenant_id=self.tenant_id,
                        subscription_plan=knowledge_rate_limit.subscription_plan,
                        operation="knowledge",
                    )
                    session.add(rate_limit_log)
                    session.commit()
                return NodeRunResult(
                    status=WorkflowNodeExecutionStatus.FAILED,
                    inputs=variables,
                    error="Sorry, you have reached the knowledge base request rate limit of your subscription.",
                    error_type="RateLimitExceeded",
                )

        # retrieve knowledge
        try:
            # results = self._fetch_dataset_retriever(node_data=node_data, query=query)
            results = self._fetch_dataset_retriever(node_data=node_data, query=query, dataset_ids=dataset_ids)
            outputs = {"result": ArrayObjectSegment(value=results)}
            return NodeRunResult(
                status=WorkflowNodeExecutionStatus.SUCCEEDED,
                inputs=variables,
                process_data=None,
                outputs=outputs,  # type: ignore
            )

        except KnowledgeRetrievalNodeError as e:
            logger.warning("Error when running knowledge retrieval node")
            return NodeRunResult(
                status=WorkflowNodeExecutionStatus.FAILED,
                inputs=variables,
                error=str(e),
                error_type=type(e).__name__,
            )
        # Temporary handle all exceptions from DatasetRetrieval class here.
        except Exception as e:
            return NodeRunResult(
                status=WorkflowNodeExecutionStatus.FAILED,
                inputs=variables,
                error=str(e),
                error_type=type(e).__name__,
            )
        finally:
            db.session.close()

    # def _fetch_dataset_retriever(self, node_data: KnowledgeRetrievalNodeData, query: str) -> list[dict[str, Any]]:
    def _fetch_dataset_retriever(self, node_data: KnowledgeRetrievalNodeData, query: str, dataset_ids: Optional[list[str]] = None) -> list[dict[str, Any]]:
        available_datasets = []
        # dataset_ids = node_data.dataset_ids
        if dataset_ids is None or dataset_ids == []:
            dataset_ids = node_data.dataset_ids

        # Subquery: Count the number of available documents for each dataset
        subquery = (
            db.session.query(Document.dataset_id, func.count(Document.id).label("available_document_count"))
            .filter(
                Document.indexing_status == "completed",
                Document.enabled == True,
                Document.archived == False,
                Document.dataset_id.in_(dataset_ids),
            )
            .group_by(Document.dataset_id)
            .having(func.count(Document.id) > 0)
            .subquery()
        )

        results = (
            db.session.query(Dataset)
            .outerjoin(subquery, Dataset.id == subquery.c.dataset_id)
            .filter(Dataset.tenant_id == self.tenant_id, Dataset.id.in_(dataset_ids))
            .filter((subquery.c.available_document_count > 0) | (Dataset.provider == "external"))
            .all()
        )

        # avoid blocking at retrieval
        db.session.close()

        for dataset in results:
            # pass if dataset is not available
            if not dataset:
                continue
            available_datasets.append(dataset)

        # Auto-merge dataset retrieval configurations if using dataset_ids_variable_selector # 如果使用dataset_ids_variable_selector，则自动合并数据集检索配置
        if (node_data.dataset_ids_variable_selector and available_datasets and 
            getattr(node_data, 'auto_merge_dataset_configs', True)):
            node_data = self._merge_dataset_retrieval_configs(node_data, available_datasets)
        
        metadata_filter_document_ids, metadata_condition = self._get_metadata_filter_condition(
            [dataset.id for dataset in available_datasets], query, node_data
        )
        all_documents = []
        dataset_retrieval = DatasetRetrieval()
        if node_data.retrieval_mode == DatasetRetrieveConfigEntity.RetrieveStrategy.SINGLE.value:
            # fetch model config
            if node_data.single_retrieval_config is None:
                raise ValueError("single_retrieval_config is required")
            model_instance, model_config = self.get_model_config(node_data.single_retrieval_config.model)
            # check model is support tool calling
            model_type_instance = model_config.provider_model_bundle.model_type_instance
            model_type_instance = cast(LargeLanguageModel, model_type_instance)
            # get model schema
            model_schema = model_type_instance.get_model_schema(
                model=model_config.model, credentials=model_config.credentials
            )

            if model_schema:
                planning_strategy = PlanningStrategy.REACT_ROUTER
                features = model_schema.features
                if features:
                    if ModelFeature.TOOL_CALL in features or ModelFeature.MULTI_TOOL_CALL in features:
                        planning_strategy = PlanningStrategy.ROUTER
                all_documents = dataset_retrieval.single_retrieve(
                    available_datasets=available_datasets,
                    tenant_id=self.tenant_id,
                    user_id=self.user_id,
                    app_id=self.app_id,
                    user_from=self.user_from.value,
                    query=query,
                    model_config=model_config,
                    model_instance=model_instance,
                    planning_strategy=planning_strategy,
                    metadata_filter_document_ids=metadata_filter_document_ids,
                    metadata_condition=metadata_condition,
                )
        elif node_data.retrieval_mode == DatasetRetrieveConfigEntity.RetrieveStrategy.MULTIPLE.value:
            if node_data.multiple_retrieval_config is None:
                raise ValueError("multiple_retrieval_config is required")
            if node_data.multiple_retrieval_config.reranking_mode == "reranking_model":
                if node_data.multiple_retrieval_config.reranking_model:
                    reranking_model = {
                        "reranking_provider_name": node_data.multiple_retrieval_config.reranking_model.provider,
                        "reranking_model_name": node_data.multiple_retrieval_config.reranking_model.model,
                    }
                else:
                    reranking_model = None
                weights = None
            elif node_data.multiple_retrieval_config.reranking_mode == "weighted_score":
                if node_data.multiple_retrieval_config.weights is None:
                    raise ValueError("weights is required")
                reranking_model = None
                vector_setting = node_data.multiple_retrieval_config.weights.vector_setting
                weights = {
                    "vector_setting": {
                        "vector_weight": vector_setting.vector_weight,
                        "embedding_provider_name": vector_setting.embedding_provider_name,
                        "embedding_model_name": vector_setting.embedding_model_name,
                    },
                    "keyword_setting": {
                        "keyword_weight": node_data.multiple_retrieval_config.weights.keyword_setting.keyword_weight
                    },
                }
            else:
                reranking_model = None
                weights = None
            all_documents = dataset_retrieval.multiple_retrieve(
                app_id=self.app_id,
                tenant_id=self.tenant_id,
                user_id=self.user_id,
                user_from=self.user_from.value,
                available_datasets=available_datasets,
                query=query,
                top_k=node_data.multiple_retrieval_config.top_k,
                score_threshold=node_data.multiple_retrieval_config.score_threshold
                if node_data.multiple_retrieval_config.score_threshold is not None
                else 0.0,
                reranking_mode=node_data.multiple_retrieval_config.reranking_mode,
                reranking_model=reranking_model,
                weights=weights,
                reranking_enable=node_data.multiple_retrieval_config.reranking_enable,
                metadata_filter_document_ids=metadata_filter_document_ids,
                metadata_condition=metadata_condition,
            )
        dify_documents = [item for item in all_documents if item.provider == "dify"]
        external_documents = [item for item in all_documents if item.provider == "external"]
        retrieval_resource_list = []
        # deal with external documents
        for item in external_documents:
            source = {
                "metadata": {
                    "_source": "knowledge",
                    "dataset_id": item.metadata.get("dataset_id"),
                    "dataset_name": item.metadata.get("dataset_name"),
                    "document_id": item.metadata.get("document_id") or item.metadata.get("title"),
                    "document_name": item.metadata.get("title"),
                    "data_source_type": "external",
                    "retriever_from": "workflow",
                    "score": item.metadata.get("score"),
                    "doc_metadata": item.metadata,
                },
                "title": item.metadata.get("title"),
                "content": item.page_content,
            }
            retrieval_resource_list.append(source)
        # deal with dify documents
        if dify_documents:
            records = RetrievalService.format_retrieval_documents(dify_documents)
            if records:
                for record in records:
                    segment = record.segment
                    dataset = db.session.query(Dataset).filter_by(id=segment.dataset_id).first()  # type: ignore
                    document = (
                        db.session.query(Document)
                        .filter(
                            Document.id == segment.document_id,
                            Document.enabled == True,
                            Document.archived == False,
                        )
                        .first()
                    )
                    if dataset and document:
                        source = {
                            "metadata": {
                                "_source": "knowledge",
                                "dataset_id": dataset.id,
                                "dataset_name": dataset.name,
                                "document_id": document.id,
                                "document_name": document.name,
                                "data_source_type": document.data_source_type,
                                "segment_id": segment.id,
                                "retriever_from": "workflow",
                                "score": record.score or 0.0,
                                "segment_hit_count": segment.hit_count,
                                "segment_word_count": segment.word_count,
                                "segment_position": segment.position,
                                "segment_index_node_hash": segment.index_node_hash,
                                "doc_metadata": document.doc_metadata,
                            },
                            "title": document.name,
                        }
                        if segment.answer:
                            source["content"] = f"question:{segment.get_sign_content()} \nanswer:{segment.answer}"
                        else:
                            source["content"] = segment.get_sign_content()
                        retrieval_resource_list.append(source)
        if retrieval_resource_list:
            retrieval_resource_list = sorted(
                retrieval_resource_list,
                key=lambda x: x["metadata"]["score"] if x["metadata"].get("score") is not None else 0.0,
                reverse=True,
            )
            for position, item in enumerate(retrieval_resource_list, start=1):
                item["metadata"]["position"] = position
        return retrieval_resource_list

    def _get_metadata_filter_condition(
        self, dataset_ids: list, query: str, node_data: KnowledgeRetrievalNodeData
    ) -> tuple[Optional[dict[str, list[str]]], Optional[MetadataCondition]]:
        document_query = db.session.query(Document).filter(
            Document.dataset_id.in_(dataset_ids),
            Document.indexing_status == "completed",
            Document.enabled == True,
            Document.archived == False,
        )
        filters = []  # type: ignore
        metadata_condition = None
        if node_data.metadata_filtering_mode == "disabled":
            return None, None
        elif node_data.metadata_filtering_mode == "automatic":
            automatic_metadata_filters = self._automatic_metadata_filter_func(dataset_ids, query, node_data)
            if automatic_metadata_filters:
                conditions = []
                for sequence, filter in enumerate(automatic_metadata_filters):
                    self._process_metadata_filter_func(
                        sequence,
                        filter.get("condition", ""),
                        filter.get("metadata_name", ""),
                        filter.get("value"),
                        filters,  # type: ignore
                    )
                    conditions.append(
                        Condition(
                            name=filter.get("metadata_name"),  # type: ignore
                            comparison_operator=filter.get("condition"),  # type: ignore
                            value=filter.get("value"),
                        )
                    )
                metadata_condition = MetadataCondition(
                    logical_operator=node_data.metadata_filtering_conditions.logical_operator
                    if node_data.metadata_filtering_conditions
                    else "or",  # type: ignore
                    conditions=conditions,
                )
        elif node_data.metadata_filtering_mode == "manual":
            if node_data.metadata_filtering_conditions:
                conditions = []
                if node_data.metadata_filtering_conditions:
                    for sequence, condition in enumerate(node_data.metadata_filtering_conditions.conditions):  # type: ignore
                        metadata_name = condition.name
                        expected_value = condition.value
                        if expected_value is not None and condition.comparison_operator not in ("empty", "not empty"):
                            if isinstance(expected_value, str):
                                expected_value = self.graph_runtime_state.variable_pool.convert_template(
                                    expected_value
                                ).value[0]
                                if expected_value.value_type == "number":  # type: ignore
                                    expected_value = expected_value.value  # type: ignore
                                elif expected_value.value_type == "string":  # type: ignore
                                    expected_value = re.sub(r"[\r\n\t]+", " ", expected_value.text).strip()  # type: ignore
                                else:
                                    raise ValueError("Invalid expected metadata value type")
                        conditions.append(
                            Condition(
                                name=metadata_name,
                                comparison_operator=condition.comparison_operator,
                                value=expected_value,
                            )
                        )
                        filters = self._process_metadata_filter_func(
                            sequence,
                            condition.comparison_operator,
                            metadata_name,
                            expected_value,
                            filters,
                        )
                metadata_condition = MetadataCondition(
                    logical_operator=node_data.metadata_filtering_conditions.logical_operator,
                    conditions=conditions,
                )
        else:
            raise ValueError("Invalid metadata filtering mode")
        if filters:
            if (
                node_data.metadata_filtering_conditions
                and node_data.metadata_filtering_conditions.logical_operator == "and"
            ):  # type: ignore
                document_query = document_query.filter(and_(*filters))
            else:
                document_query = document_query.filter(or_(*filters))
        documents = document_query.all()
        # group by dataset_id
        metadata_filter_document_ids = defaultdict(list) if documents else None  # type: ignore
        for document in documents:
            metadata_filter_document_ids[document.dataset_id].append(document.id)  # type: ignore
        return metadata_filter_document_ids, metadata_condition

    def _automatic_metadata_filter_func(
        self, dataset_ids: list, query: str, node_data: KnowledgeRetrievalNodeData
    ) -> list[dict[str, Any]]:
        # get all metadata field
        metadata_fields = db.session.query(DatasetMetadata).filter(DatasetMetadata.dataset_id.in_(dataset_ids)).all()
        all_metadata_fields = [metadata_field.name for metadata_field in metadata_fields]
        # get metadata model config
        metadata_model_config = node_data.metadata_model_config
        if metadata_model_config is None:
            raise ValueError("metadata_model_config is required")
        # get metadata model instance
        # fetch model config
        model_instance, model_config = self.get_model_config(metadata_model_config)
        # fetch prompt messages
        prompt_template = self._get_prompt_template(
            node_data=node_data,
            metadata_fields=all_metadata_fields,
            query=query or "",
        )
        prompt_messages, stop = self._fetch_prompt_messages(
            prompt_template=prompt_template,
            sys_query=query,
            memory=None,
            model_config=model_config,
            sys_files=[],
            vision_enabled=node_data.vision.enabled,
            vision_detail=node_data.vision.configs.detail,
            variable_pool=self.graph_runtime_state.variable_pool,
            jinja2_variables=[],
        )

        result_text = ""
        try:
            # handle invoke result
            generator = self._invoke_llm(
                node_data_model=node_data.metadata_model_config,  # type: ignore
                model_instance=model_instance,
                prompt_messages=prompt_messages,
                stop=stop,
            )

            for event in generator:
                if isinstance(event, ModelInvokeCompletedEvent):
                    result_text = event.text
                    break

            result_text_json = parse_and_check_json_markdown(result_text, [])
            automatic_metadata_filters = []
            if "metadata_map" in result_text_json:
                metadata_map = result_text_json["metadata_map"]
                for item in metadata_map:
                    if item.get("metadata_field_name") in all_metadata_fields:
                        automatic_metadata_filters.append(
                            {
                                "metadata_name": item.get("metadata_field_name"),
                                "value": item.get("metadata_field_value"),
                                "condition": item.get("comparison_operator"),
                            }
                        )
        except Exception as e:
            return []
        return automatic_metadata_filters

    def _process_metadata_filter_func(
        self, sequence: int, condition: str, metadata_name: str, value: Optional[Any], filters: list
    ):
        if value is None:
            return

        key = f"{metadata_name}_{sequence}"
        key_value = f"{metadata_name}_{sequence}_value"
        match condition:
            case "contains":
                filters.append(
                    (text(f"documents.doc_metadata ->> :{key} LIKE :{key_value}")).params(
                        **{key: metadata_name, key_value: f"%{value}%"}
                    )
                )
            case "not contains":
                filters.append(
                    (text(f"documents.doc_metadata ->> :{key} NOT LIKE :{key_value}")).params(
                        **{key: metadata_name, key_value: f"%{value}%"}
                    )
                )
            case "start with":
                filters.append(
                    (text(f"documents.doc_metadata ->> :{key} LIKE :{key_value}")).params(
                        **{key: metadata_name, key_value: f"{value}%"}
                    )
                )
            case "end with":
                filters.append(
                    (text(f"documents.doc_metadata ->> :{key} LIKE :{key_value}")).params(
                        **{key: metadata_name, key_value: f"%{value}"}
                    )
                )
            case "=" | "is":
                if isinstance(value, str):
                    filters.append(Document.doc_metadata[metadata_name] == f'"{value}"')
                else:
                    filters.append(sqlalchemy_cast(Document.doc_metadata[metadata_name].astext, Float) == value)
            case "is not" | "≠":
                if isinstance(value, str):
                    filters.append(Document.doc_metadata[metadata_name] != f'"{value}"')
                else:
                    filters.append(sqlalchemy_cast(Document.doc_metadata[metadata_name].astext, Float) != value)
            case "empty":
                filters.append(Document.doc_metadata[metadata_name].is_(None))
            case "not empty":
                filters.append(Document.doc_metadata[metadata_name].isnot(None))
            case "before" | "<":
                filters.append(sqlalchemy_cast(Document.doc_metadata[metadata_name].astext, Float) < value)
            case "after" | ">":
                filters.append(sqlalchemy_cast(Document.doc_metadata[metadata_name].astext, Float) > value)
            case "≤" | "<=":
                filters.append(sqlalchemy_cast(Document.doc_metadata[metadata_name].astext, Float) <= value)
            case "≥" | ">=":
                filters.append(sqlalchemy_cast(Document.doc_metadata[metadata_name].astext, Float) >= value)
            case _:
                pass
        return filters

    @classmethod
    def _extract_variable_selector_to_variable_mapping(
        cls,
        *,
        graph_config: Mapping[str, Any],
        node_id: str,
        node_data: KnowledgeRetrievalNodeData,  # type: ignore
    ) -> Mapping[str, Sequence[str]]:
        """
        Extract variable selector to variable mapping
        :param graph_config: graph config
        :param node_id: node id
        :param node_data: node data
        :return:
        """
        variable_mapping = {}
        variable_mapping[node_id + ".query"] = node_data.query_variable_selector
        if node_data.dataset_ids_variable_selector:
            variable_mapping[node_id + ".dataset_ids"] = node_data.dataset_ids_variable_selector
        if hasattr(node_data, 'dataset_source_mode'):
            variable_mapping[node_id + ".dataset_source_mode"] = getattr(node_data, 'dataset_source_mode', 'manual')
        return variable_mapping

    def get_model_config(self, model: ModelConfig) -> tuple[ModelInstance, ModelConfigWithCredentialsEntity]:
        model_name = model.name
        provider_name = model.provider

        model_manager = ModelManager()
        model_instance = model_manager.get_model_instance(
            tenant_id=self.tenant_id, model_type=ModelType.LLM, provider=provider_name, model=model_name
        )

        provider_model_bundle = model_instance.provider_model_bundle
        model_type_instance = model_instance.model_type_instance
        model_type_instance = cast(LargeLanguageModel, model_type_instance)

        model_credentials = model_instance.credentials

        # check model
        provider_model = provider_model_bundle.configuration.get_provider_model(
            model=model_name, model_type=ModelType.LLM
        )

        if provider_model is None:
            raise ModelNotExistError(f"Model {model_name} not exist.")

        if provider_model.status == ModelStatus.NO_CONFIGURE:
            raise ModelCredentialsNotInitializedError(f"Model {model_name} credentials is not initialized.")
        elif provider_model.status == ModelStatus.NO_PERMISSION:
            raise ModelNotSupportedError(f"Dify Hosted OpenAI {model_name} currently not support.")
        elif provider_model.status == ModelStatus.QUOTA_EXCEEDED:
            raise ModelQuotaExceededError(f"Model provider {provider_name} quota exceeded.")

        # model config
        completion_params = model.completion_params
        stop = []
        if "stop" in completion_params:
            stop = completion_params["stop"]
            del completion_params["stop"]

        # get model mode
        model_mode = model.mode
        if not model_mode:
            raise ModelNotExistError("LLM mode is required.")

        model_schema = model_type_instance.get_model_schema(model_name, model_credentials)

        if not model_schema:
            raise ModelNotExistError(f"Model {model_name} not exist.")

        return model_instance, ModelConfigWithCredentialsEntity(
            provider=provider_name,
            model=model_name,
            model_schema=model_schema,
            mode=model_mode,
            provider_model_bundle=provider_model_bundle,
            credentials=model_credentials,
            parameters=completion_params,
            stop=stop,
        )

    def _get_prompt_template(self, node_data: KnowledgeRetrievalNodeData, metadata_fields: list, query: str):
        model_mode = ModelMode.value_of(node_data.metadata_model_config.mode)  # type: ignore
        input_text = query

        prompt_messages: list[LLMNodeChatModelMessage] = []
        if model_mode == ModelMode.CHAT:
            system_prompt_messages = LLMNodeChatModelMessage(
                role=PromptMessageRole.SYSTEM, text=METADATA_FILTER_SYSTEM_PROMPT
            )
            prompt_messages.append(system_prompt_messages)
            user_prompt_message_1 = LLMNodeChatModelMessage(
                role=PromptMessageRole.USER, text=METADATA_FILTER_USER_PROMPT_1
            )
            prompt_messages.append(user_prompt_message_1)
            assistant_prompt_message_1 = LLMNodeChatModelMessage(
                role=PromptMessageRole.ASSISTANT, text=METADATA_FILTER_ASSISTANT_PROMPT_1
            )
            prompt_messages.append(assistant_prompt_message_1)
            user_prompt_message_2 = LLMNodeChatModelMessage(
                role=PromptMessageRole.USER, text=METADATA_FILTER_USER_PROMPT_2
            )
            prompt_messages.append(user_prompt_message_2)
            assistant_prompt_message_2 = LLMNodeChatModelMessage(
                role=PromptMessageRole.ASSISTANT, text=METADATA_FILTER_ASSISTANT_PROMPT_2
            )
            prompt_messages.append(assistant_prompt_message_2)
            user_prompt_message_3 = LLMNodeChatModelMessage(
                role=PromptMessageRole.USER,
                text=METADATA_FILTER_USER_PROMPT_3.format(
                    input_text=input_text,
                    metadata_fields=json.dumps(metadata_fields, ensure_ascii=False),
                ),
            )
            prompt_messages.append(user_prompt_message_3)
            return prompt_messages
        elif model_mode == ModelMode.COMPLETION:
            return LLMNodeCompletionModelPromptTemplate(
                text=METADATA_FILTER_COMPLETION_PROMPT.format(
                    input_text=input_text,
                    metadata_fields=json.dumps(metadata_fields, ensure_ascii=False),
                )
            )

        else:
            raise InvalidModelTypeError(f"Model mode {model_mode} not support.")
        
    def _merge_dataset_retrieval_configs(self, node_data: KnowledgeRetrievalNodeData, available_datasets: list[Dataset]) -> KnowledgeRetrievalNodeData:
        """
        Merge retrieval configurations from datasets when using dynamic dataset IDs.
        This method analyzes the retrieval configurations stored in each dataset and
        creates an optimal merged configuration for the knowledge retrieval node.
        """
        from copy import deepcopy
        
        # Create a copy of node_data to avoid modifying the original # 创建节点数据的副本以避免修改原始副本
        merged_node_data = deepcopy(node_data)
        
        # Extract retrieval configurations from datasets # 从数据集中提取检索配置
        dataset_configs = []
        for dataset in available_datasets:
            if dataset.retrieval_model:
                dataset_configs.append(dataset.retrieval_model_dict)
        
        # If no dataset configs found, use current node configuration # 如果没有找到数据集配置，则使用当前节点配置
        if not dataset_configs:
            return merged_node_data
        
        # Determine the best retrieval mode based on dataset configurations # 根据数据集配置确定最佳检索模式
        # Priority: multiple > single (multiple is more comprehensive) # 优先级：multiple > single（multiple更全面）
        has_multiple_config = any(self._is_multiple_retrieval_config(config) for config in dataset_configs)
        
        if has_multiple_config:
            # Merge multiple retrieval configurations # 合并多个检索配置
            merged_node_data.retrieval_mode = "multiple"
            merged_node_data.multiple_retrieval_config = self._merge_multiple_retrieval_configs(dataset_configs)
            merged_node_data.single_retrieval_config = None
        else:
            # Use single retrieval configuration from first dataset # 使用第一个数据集的单检索配置
            merged_node_data.retrieval_mode = "single"
            merged_node_data.single_retrieval_config = self._create_single_retrieval_config(dataset_configs[0])
            merged_node_data.multiple_retrieval_config = None
        
        return merged_node_data
    
    def _is_multiple_retrieval_config(self, config: dict) -> bool:
        """Check if a dataset configuration is suitable for multiple retrieval mode."""
        # Check if configuration has multiple retrieval specific fields # 检查配置是否具有多个检索特定字段
        multiple_fields = ['search_method', 'reranking_enable', 'reranking_model', 'weights']
        return any(field in config for field in multiple_fields)
    
    def _merge_multiple_retrieval_configs(self, dataset_configs: list[dict]) -> 'MultipleRetrievalConfig':
        """Merge multiple retrieval configurations from datasets."""
        from .entities import MultipleRetrievalConfig, RerankingModelConfig, WeightedScoreConfig, VectorSetting, KeywordSetting
        
        # Use the first config as base and merge others # 使用第一个配置作为基础并合并其他配置
        base_config = dataset_configs[0]
        
        # Calculate average top_k # 计算平均top_k
        top_k_values = [config.get('top_k', 2) for config in dataset_configs if config.get('top_k')]
        avg_top_k = int(sum(top_k_values) / len(top_k_values)) if top_k_values else 2
        
        # Use minimum score_threshold for better recall # 使用最小score_threshold以获得更好的召回率
        score_thresholds = [config.get('score_threshold', 0.0) for config in dataset_configs 
                          if config.get('score_threshold_enabled', False)]
        min_score_threshold = min(score_thresholds) if score_thresholds else None
        
        # Check if any dataset has reranking enabled # 检查是否有任何数据集启用了重新排名
        reranking_enabled = any(config.get('reranking_enable', False) for config in dataset_configs)
        
        # Get reranking model from first dataset that has it # 从具有它的第一个数据集中获取重新排名模型
        reranking_model = None
        for config in dataset_configs:
            if config.get('reranking_model') and config['reranking_model'].get('reranking_provider_name'):
                reranking_model = RerankingModelConfig(
                    provider=config['reranking_model']['reranking_provider_name'],
                    model=config['reranking_model']['reranking_model_name']
                )
                break
        
        return MultipleRetrievalConfig(
            top_k=avg_top_k,
            score_threshold=min_score_threshold,
            reranking_mode="reranking_model" if reranking_model else "weighted_score",
            reranking_enable=reranking_enabled,
            reranking_model=reranking_model,
            weights=None  # Could be enhanced to merge weights from datasets # 可以增强以从数据集中合并权重
        )
    
    def _create_single_retrieval_config(self, dataset_config: dict) -> 'SingleRetrievalConfig':
        """Create single retrieval configuration from dataset config."""
        from .entities import SingleRetrievalConfig, ModelConfig
        
        # Use default model config if not specified in dataset # 如果在数据集中未指定，则使用默认模型配置
        # This could be enhanced to read model config from dataset # 这可以增强为从数据集读取模型配置
        default_model = ModelConfig(
            provider="openai",  # Default provider
            name="gpt-3.5-turbo",  # Default model
            mode="chat",
            completion_params={}
        )
        
        return SingleRetrievalConfig(model=default_model)
