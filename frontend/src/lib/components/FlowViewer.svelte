<script lang="ts">
	import type { FlowValue } from '$lib/gen'
	import { Tab, Tabs, TabContent } from './common'
	import SchemaViewer from './SchemaViewer.svelte'
	import FlowGraphViewer from './FlowGraphViewer.svelte'

	import HighlightTheme from './HighlightTheme.svelte'
	import FlowViewerInner from './FlowViewerInner.svelte'
	import FlowInputViewer from './FlowInputViewer.svelte'

	interface Props {
		flow: {
			summary: string
			description?: string
			value: FlowValue
			schema?: any
		}
		initialOpen?: number | undefined
		noSide?: boolean
		noGraph?: boolean
		tab?: 'ui' | 'raw' | 'schema'
		noSummary?: boolean
		noGraphDownload?: boolean
	}

	let {
		flow,
		initialOpen = undefined,
		noSide = false,
		noGraph = false,
		tab = $bindable(noGraph ? 'schema' : 'ui'),
		noSummary = false,
		noGraphDownload = false
	}: Props = $props()

	let open: { [id: number]: boolean } = {}
	if (initialOpen) {
		open[initialOpen] = true
	}
</script>

<HighlightTheme />

<Tabs bind:selected={tab}>
	{#if !noGraph}
		<Tab value="ui">Graph</Tab>
	{/if}
	<Tab value="raw">Raw</Tab>
	<Tab value="schema">Input Schema</Tab>

	{#snippet content()}
		<TabContent value="ui">
			<div class="flow-root w-full pb-4">
				{#if !noSummary}
					<h2 class="my-4">{flow.summary}</h2>
					<div>{flow.description ?? ''}</div>
				{/if}

				<p class="font-black text-lg w-full my-4">
					<span>Flow Input</span>
				</p>
				{#if flow.schema && flow.schema.properties && Object.keys(flow.schema.properties).length > 0 && flow.schema}
					<FlowInputViewer schema={flow.schema} />
				{:else}
					<div class="text-secondary text-xs italic mb-4">No inputs</div>
				{/if}

				<FlowGraphViewer download={!noGraphDownload} {noSide} {flow} overflowAuto />
			</div>
		</TabContent>
		<TabContent value="raw">
			<FlowViewerInner {flow} />
		</TabContent>
		<TabContent value="schema">
			<div class="my-4"></div>
			<SchemaViewer schema={flow.schema} />
		</TabContent>
	{/snippet}
</Tabs>
