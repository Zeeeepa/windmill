<script lang="ts">
	import { getContext, onDestroy } from 'svelte'
	import { initConfig, initOutput } from '../../editor/appUtils'
	import type {
		AppViewerContext,
		ComponentCustomCSS,
		ListContext,
		ListInputs,
		RichConfigurations
	} from '../../types'
	import { initCss } from '../../utils'
	import AlignWrapper from '../helpers/AlignWrapper.svelte'
	import InitializeComponent from '../helpers/InitializeComponent.svelte'
	import { components } from '../../editor/component'
	import ResolveConfig from '../helpers/ResolveConfig.svelte'
	import ResolveStyle from '../helpers/ResolveStyle.svelte'
	import { twMerge } from 'tailwind-merge'
	import LightweightResourcePicker from '$lib/components/LightweightResourcePicker.svelte'

	interface Props {
		id: string
		configuration: RichConfigurations
		verticalAlignment?: 'top' | 'center' | 'bottom' | undefined
		customCss?: ComponentCustomCSS<'userresourcecomponent'> | undefined
		render: boolean
	}

	let {
		id,
		configuration,
		verticalAlignment = undefined,
		customCss = undefined,
		render
	}: Props = $props()

	const { app, worldStore, componentControl } = getContext<AppViewerContext>('AppViewerContext')

	let resolvedConfig = $state(
		initConfig(components['userresourcecomponent'].initialData.configuration, configuration)
	)

	const iterContext = getContext<ListContext>('ListWrapperContext')
	const listInputs: ListInputs | undefined = getContext<ListInputs>('ListInputs')

	let outputs = initOutput($worldStore, id, {
		result: undefined as string | undefined
	})

	let css = $state(initCss($app.css?.['userresourcecomponent'], customCss))

	let classInput = $derived(
		twMerge(
			'windmillapp w-full px-2',
			css?.input?.class ?? '',
			'wm-input',
			'wm-user-resource-select'
		)
	)

	let value: string | undefined = $state(outputs.result.peak()?.replace('$res:', ''))

	value && assignValue(outputs.result.peak())

	onDestroy(() => {
		listInputs?.remove(id)
	})

	$componentControl[id] = {
		setValue(nvalue: string) {
			value = nvalue
		},
		askNewResource() {
			resourcePicker?.askNewResource()
		}
	}

	function assignValue(value: string) {
		let nval
		if (!value || value === '') {
			nval = undefined
		} else {
			nval = '$res:' + value.replace('$res:', '')
		}
		outputs?.result.set(nval)
		if (iterContext && listInputs) {
			listInputs.set(id, nval)
		}
	}

	let resourcePicker: LightweightResourcePicker | undefined = $state(undefined)
</script>

{#each Object.keys(components['userresourcecomponent'].initialData.configuration) as key (key)}
	<ResolveConfig
		{id}
		{key}
		bind:resolvedConfig={resolvedConfig[key]}
		configuration={configuration[key]}
	/>
{/each}

{#each Object.keys(css ?? {}) as key (key)}
	<ResolveStyle
		{id}
		{customCss}
		{key}
		bind:css={css[key]}
		componentStyle={$app.css?.textinputcomponent}
	/>
{/each}

<InitializeComponent {id} />
{#if render}
	<AlignWrapper {render} {verticalAlignment}>
		<div
			class="relative w-full {classInput}',
"
		>
			<LightweightResourcePicker
				expressOAuthSetup={resolvedConfig.expressOauthSetup}
				bind:this={resourcePicker}
				{value}
				on:change={(e) => {
					assignValue(e.detail)
				}}
				disabled={resolvedConfig.disabled}
				resourceType={resolvedConfig.resourceType}
			/>
		</div>
	</AlignWrapper>
{/if}
