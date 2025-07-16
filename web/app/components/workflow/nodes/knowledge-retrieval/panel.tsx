import type { FC } from 'react'
import {
  memo,
  useCallback,
  useMemo,
} from 'react'
import { intersectionBy } from 'lodash-es'
import { useTranslation } from 'react-i18next'
import Switch from '@/app/components/base/switch'
import VarReferencePicker from '../_base/components/variable/var-reference-picker'
import useConfig from './use-config'
import RetrievalConfig from './components/retrieval-config'
import AddKnowledge from './components/add-dataset'
import DatasetList from './components/dataset-list'
import MetadataFilter from './components/metadata/metadata-filter'
import type { KnowledgeRetrievalNodeType } from './types'
import Field from '@/app/components/workflow/nodes/_base/components/field'
import Split from '@/app/components/workflow/nodes/_base/components/split'
import OutputVars, { VarItem } from '@/app/components/workflow/nodes/_base/components/output-vars'
import type { NodePanelProps } from '@/app/components/workflow/types'
import Tooltip from '@/app/components/base/tooltip'
import { RiQuestionLine } from '@remixicon/react'

const i18nPrefix = 'workflow.nodes.knowledgeRetrieval'

const Panel: FC<NodePanelProps<KnowledgeRetrievalNodeType>> = ({
  id,
  data,
}) => {
  const { t } = useTranslation()

  const {
    readOnly,
    inputs,
    handleQueryVarChange,
    filterVar,
    handleModelChanged,
    handleCompletionParamsChange,
    handleRetrievalModeChange,
    handleMultipleRetrievalConfigChange,
    selectedDatasets,
    selectedDatasetsLoaded,
    handleOnDatasetsChange,
    rerankModelOpen,
    setRerankModelOpen,
    handleAddCondition,
    handleMetadataFilterModeChange,
    handleRemoveCondition,
    handleToggleConditionLogicalOperator,
    handleUpdateCondition,
    handleMetadataModelChange,
    handleMetadataCompletionParamsChange,
    availableStringVars,
    availableStringNodesWithParent,
    availableNumberVars,
    availableNumberNodesWithParent,
    handleDatasetSourceModeChange,
    handleDatasetIdsVarChange,
    handleAutoMergeConfigsChange,
    filterDatasetVar,
  } = useConfig(id, data)

  const handleOpenFromPropsChange = useCallback((openFromProps: boolean) => {
    setRerankModelOpen(openFromProps)
  }, [setRerankModelOpen])

  const metadataList = useMemo(() => {
    return intersectionBy(...selectedDatasets.filter((dataset) => {
      return !!dataset.doc_metadata
    }).map((dataset) => {
      return dataset.doc_metadata!
    }), 'name')
  }, [selectedDatasets])

  return (
    <div className='pt-2'>
      <div className='space-y-4 px-4 pb-2'>
        {/* {JSON.stringify(inputs, null, 2)} */}
        <Field
          title={t(`${i18nPrefix}.queryVariable`)}
          required
        >
          <VarReferencePicker
            nodeId={id}
            readonly={readOnly}
            isShowNodeName
            value={inputs.query_variable_selector}
            onChange={handleQueryVarChange}
            filterVar={filterVar}
          />
        </Field>

        <Field
          title={t(`${i18nPrefix}.knowledge`)}
          required
          operations={
            <div className='flex shrink-0 items-center'>
              <div className='system-xs-medium-uppercase mr-0.5 text-text-tertiary'>
                {/* {t(`${i18nPrefix}.dynamicsKnowledge`)} */}
              {inputs.dataset_source_mode === 'manual'
                ? t(`${i18nPrefix}.manualDatasetSelection`)
                : t(`${i18nPrefix}.variableDatasetSelection`)
              }
              </div>
              <Tooltip popupContent={
                <div className='max-w-[150px]'>
                  {/* {t(`${i18nPrefix}.contextTooltip`)!} */}
                  {t(`${i18nPrefix}.datasetVariableDesc`)}
                </div>
              }>
                <div>
                  <RiQuestionLine className='size-3.5 text-text-quaternary' />
                </div>
              </Tooltip>
              <Switch
                className='ml-2 mr-2'
                defaultValue={inputs.dataset_source_mode === 'variable'}
                onChange={checked => handleDatasetSourceModeChange(checked ? 'variable' : 'manual')}
                size='md'
                disabled={readOnly}
              />
            <div className='flex items-center space-x-1'>
              <RetrievalConfig
                payload={{
                  retrieval_mode: inputs.retrieval_mode,
                  multiple_retrieval_config: inputs.multiple_retrieval_config,
                  single_retrieval_config: inputs.single_retrieval_config,
                }}
                onRetrievalModeChange={handleRetrievalModeChange}
                onMultipleRetrievalConfigChange={handleMultipleRetrievalConfigChange}
                singleRetrievalModelConfig={inputs.single_retrieval_config?.model}
                onSingleRetrievalModelChange={handleModelChanged as any}
                onSingleRetrievalModelParamsChange={handleCompletionParamsChange}
                readonly={readOnly || (inputs.dataset_source_mode === 'variable' && inputs.auto_merge_dataset_configs !== false) || (inputs.dataset_source_mode === 'manual' && !selectedDatasets.length)}
                openFromProps={rerankModelOpen}
                onOpenFromPropsChange={handleOpenFromPropsChange}
                selectedDatasets={selectedDatasets}
                variableMode={inputs.dataset_source_mode === 'variable'}
                autoMergeConfigs={inputs.auto_merge_dataset_configs !== false}
              />
              {inputs.dataset_source_mode === 'variable' && (
                  <div className='flex items-center space-x-1'>
                    <div className='h-3 w-px bg-divider-regular'></div>
                    <div className='system-2xs-medium-uppercase flex h-[18px] items-center rounded-[5px] border border-divider-deep px-1 capitalize text-text-tertiary'>
                      {inputs.auto_merge_dataset_configs !== false ? 'Auto Config' : 'Manual Config'}
                    </div>
                    <Tooltip popupContent={
                      <div className='max-w-[200px]'>
                        {t(`${i18nPrefix}.nodes.knowledgeRetrieval.autoMergeTooltip`)}
                      </div>
                    }>
                      <div>
                        <RiQuestionLine className='size-3.5 text-text-quaternary' />
                      </div>
                    </Tooltip>
                    <Switch
                      className='ml-1'
                      defaultValue={inputs.auto_merge_dataset_configs !== false}
                      onChange={handleAutoMergeConfigsChange}
                      size='sm'
                      disabled={readOnly}
                    />
                  </div>
                )}
                {inputs.dataset_source_mode === 'manual' && !readOnly && (
                  <>
                    <div className='h-3 w-px bg-divider-regular'></div>
                    <AddKnowledge
                      selectedIds={inputs.dataset_ids}
                      onChange={handleOnDatasetsChange}
                    />
                  </>
              )}
            </div>
            </div>
          }
        >
          {(inputs.dataset_source_mode === 'variable')
            ? (
              <VarReferencePicker
              nodeId={id}
              readonly={readOnly}
              isShowNodeName
              value={inputs.dataset_ids_variable_selector || []}
              onChange={handleDatasetIdsVarChange}
              filterVar={filterDatasetVar}
            />
            )
            : (
              <DatasetList
                list={selectedDatasets}
                onChange={handleOnDatasetsChange}
                readonly={readOnly}
              />
            )}
        </Field>
      </div>
      <div className='mb-2 py-2'>
        <MetadataFilter
          metadataList={metadataList}
          selectedDatasetsLoaded={selectedDatasetsLoaded}
          metadataFilterMode={inputs.metadata_filtering_mode}
          metadataFilteringConditions={inputs.metadata_filtering_conditions}
          handleAddCondition={handleAddCondition}
          handleMetadataFilterModeChange={handleMetadataFilterModeChange}
          handleRemoveCondition={handleRemoveCondition}
          handleToggleConditionLogicalOperator={handleToggleConditionLogicalOperator}
          handleUpdateCondition={handleUpdateCondition}
          metadataModelConfig={inputs.metadata_model_config}
          handleMetadataModelChange={handleMetadataModelChange}
          handleMetadataCompletionParamsChange={handleMetadataCompletionParamsChange}
          availableStringVars={availableStringVars}
          availableStringNodesWithParent={availableStringNodesWithParent}
          availableNumberVars={availableNumberVars}
          availableNumberNodesWithParent={availableNumberNodesWithParent}
        />
      </div>
      <Split />
      <div>
        <OutputVars>
          <>
            <VarItem
              name='result'
              type='Array[Object]'
              description={t(`${i18nPrefix}.outputVars.output`)}
              subItems={[
                {
                  name: 'content',
                  type: 'string',
                  description: t(`${i18nPrefix}.outputVars.content`),
                },
                // url, title, link like bing search reference result: link, link page title, link page icon
                {
                  name: 'title',
                  type: 'string',
                  description: t(`${i18nPrefix}.outputVars.title`),
                },
                {
                  name: 'url',
                  type: 'string',
                  description: t(`${i18nPrefix}.outputVars.url`),
                },
                {
                  name: 'icon',
                  type: 'string',
                  description: t(`${i18nPrefix}.outputVars.icon`),
                },
                {
                  name: 'metadata',
                  type: 'object',
                  description: t(`${i18nPrefix}.outputVars.metadata`),
                },
              ]}
            />

          </>
        </OutputVars>
      </div>
    </div>
  )
}

export default memo(Panel)
