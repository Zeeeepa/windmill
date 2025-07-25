<script lang="ts">
	import { Badge, type AlertType } from '../common'
	import Popover from '$lib/components/meltComponents/Popover.svelte'
	import Toggle from '$lib/components/Toggle.svelte'
	import SimpleEditor from '../SimpleEditor.svelte'
	import Label from '../Label.svelte'
	import Section from '../Section.svelte'
	import Tooltip from '../Tooltip.svelte'
	import Button from '../common/button/Button.svelte'
	import { twMerge } from 'tailwind-merge'
	import { ColumnIdentity, type ColumnDef } from '../apps/components/display/dbtable/utils'
	import { offset, flip, shift } from 'svelte-floating-ui/dom'

	import Alert from '../common/alert/Alert.svelte'
	interface Props {
		value: ColumnDef | undefined
		trigger?: import('svelte').Snippet
	}

	let { value = $bindable(), trigger: trigger_render }: Props = $props()

	const presets = [
		{
			label: 'None',
			value: null
		},
		{
			label: 'Currency CHF',
			value: 'value + " CHF"'
		},
		{
			label: 'Currency USD',
			value: '"$ " + value'
		},
		{
			label: 'Date',
			value: 'new Date(value).toLocaleDateString()'
		},
		{
			label: 'Percentage',
			value: 'value + " %"'
		},
		{
			label: 'Currency GBP',
			value: 'value + " £"'
		},
		{
			label: 'Currency EUR',
			value: 'value + " €"'
		},
		{
			label: 'Currency JPY',
			value: 'value + " ¥"'
		},
		{
			label: 'Decimal places (2)',
			value: 'parseFloat(value).toFixed(2)'
		},
		{
			label: 'Uppercase',
			value: 'value.toUpperCase()'
		},
		{
			label: 'Lowercase',
			value: 'value.toLowerCase()'
		},
		{
			label: 'Boolean (True/False)',
			value: 'value ? "True" : "False"'
		},

		{
			label: 'Object',
			value: 'JSON.stringify(value, null, 2)'
		}
	]

	let renderCount = $state(0)

	function computeWarning(columnMetadata, value) {
		if (columnMetadata?.isnullable === 'NO' && !columnMetadata?.defaultvalue) {
			if ([ColumnIdentity.ByDefault].includes(columnMetadata?.isidentity)) {
				return {
					type: 'info' as AlertType,
					title: 'Value will be generated',
					message:
						'The column is an identity column. The value will be generated by the database unless a default is provided.'
				}
			}
			if (value?.hideInsert && columnMetadata?.isidentity !== ColumnIdentity.Always) {
				return {
					type: 'warning' as AlertType,
					title: 'No default value',
					message:
						"The column is not nullable and doesn't have a default value. A default value is required."
				}
			}
		}

		if (columnMetadata?.defaultvalue !== null) {
			return {
				type: 'info' as AlertType,
				title: 'Default value',
				message: `${
					value?.hideInsert ? '' : 'You may want to hide this field from insert. '
				}The column has a default value defined in the database. The default value is: ${
					value?.defaultvalue
				}`
			}
		}

		if (columnMetadata?.isnullable === 'YES') {
			return {
				type: 'info' as AlertType,
				title: 'Default value',
				message: `${
					value?.hideInsert ? '' : 'You may want to hide this field from insert. '
				}The column can be null. If no value is provided, the default value will be null.`
			}
		}

		return null
	}

	let warning = $derived(computeWarning(value, value))
</script>

<Popover
	floatingConfig={{
		strategy: 'fixed',
		placement: 'left-start',
		middleware: [offset(8), flip(), shift()]
	}}
	contentClasses="max-h-[70vh] overflow-y-auto p-4 flex flex-col gap-4 w-96"
	closeOnOtherPopoverOpen
>
	{#snippet trigger()}
		{@render trigger_render?.()}
	{/snippet}

	{#snippet content()}
		{#if value}
			<Section label="Column settings">
				{#snippet header()}
					<Badge color="blue">
						{value.field}
					</Badge>
				{/snippet}
				<Label label="Skip for select and update">
					{#snippet header()}
						<Tooltip>
							By default, all columns are included in the select and update queries. If you want to
							exclude a column from the select and update queries, you can set this property to
							true.
						</Tooltip>
					{/snippet}
					{#snippet action()}
						<Toggle
							on:pointerdown={(e) => {
								e?.stopPropagation()
							}}
							bind:checked={value.ignored}
							size="xs"
							disabled={value?.isprimarykey}
						/>
					{/snippet}
					{#if value?.isprimarykey}
						<Alert type="warning" size="xs" title="Primary key" class="my-1">
							You cannot skip a primary key.
						</Alert>
					{/if}
				</Label>

				<Label label="Hide from insert">
					{#snippet header()}
						<Tooltip>
							By default, all columns are used to generate the submit form. If you want to exclude a
							column from the submit form, you can set this property to true. If the column is not
							nullable or doesn't have a default value, a default value will be required.
						</Tooltip>
					{/snippet}
					{#snippet action()}
						<Toggle
							disabled={value?.isidentity === ColumnIdentity.Always}
							on:pointerdown={(e) => {
								e?.stopPropagation()
							}}
							bind:checked={value.hideInsert}
							size="xs"
						/>
					{/snippet}
				</Label>
				{#if value?.isidentity === ColumnIdentity.Always}
					<Alert type="warning" size="xs" title="Identity column" class="my-1">
						This column is an ALWAYS identity column and so can't be provided by the user.
					</Alert>
				{/if}

				{#if warning}
					<Alert type={warning.type} size="xs" title={warning.title} class="my-2">
						{warning.message}
					</Alert>
				{/if}

				{#if value?.defaultvalue !== null && value?.hideInsert}
					<Toggle
						bind:checked={value.overrideDefaultValue}
						size="xs"
						options={{
							right: `Override default value: ${value?.defaultvalue}`
						}}
						on:change={() => {
							if (!value || !value.overrideDefaultValue) {
								if (value) {
									value.defaultValueNull = false
									value.defaultUserValue = undefined
								}
							}
						}}
					/>
				{/if}
				<Label label="Default input">
					{#snippet header()}
						<Tooltip>
							By default, all columns are used to generate the submit form. If you want to exclude a
							column from the submit form, you can set this property to true. If the column is not
							nullable or doesn't have a default value, a default value will be required.
						</Tooltip>
					{/snippet}
					{#if value?.datatype}
						{@const type = value?.datatype}

						<div class="flex flex-row items-center gap-2">
							<Badge color="dark-gray">
								Type:
								{type}
							</Badge>

							{#if value?.isnullable == 'YES' && value.hideInsert}
								<Toggle
									bind:checked={value.defaultValueNull}
									size="xs"
									options={{
										right: 'Set to null'
									}}
									disabled={value.hideInsert &&
										value?.defaultvalue !== null &&
										!value?.overrideDefaultValue}
									on:change={() => {
										if (value?.defaultValueNull && value) {
											value.defaultUserValue = null
										}
									}}
								/>
							{/if}
						</div>

						<input
							type="text"
							placeholder="Default value"
							class="mt-2"
							bind:value={value.defaultUserValue}
							disabled={value.defaultValueNull ||
								(value.hideInsert && value?.defaultvalue !== null && !value?.overrideDefaultValue)}
						/>
					{/if}
				</Label>
			</Section>

			<Section label="AG Grid configuration">
				<div
					class={twMerge('flex flex-col gap-4', value.ignored ? 'opacity-50 cursor-none ' : '')}
					onpointerdown={(e) => {
						if (value?.ignored) {
							e?.stopPropagation()
						}
					}}
				>
					<Label label="Header name">
						<input type="text" placeholder="Header name" bind:value={value.headerName} />
					</Label>

					<Label label="Editable value">
						<Toggle
							on:pointerdown={(e) => {
								e?.stopPropagation()
							}}
							options={{ right: 'Editable' }}
							bind:checked={value.editable}
							size="xs"
						/>
					</Label>

					<Label label="Min width (px)">
						<input type="number" placeholder="width" bind:value={value.minWidth} />
					</Label>

					<Label label="Flex">
						{#snippet header()}
							<Tooltip
								documentationLink="https://www.ag-grid.com/javascript-data-grid/column-sizing/#column-flex"
							>
								It's often required that one or more columns fill the entire available space in the
								grid. For this scenario, it is possible to use the flex config. Some columns could
								be set with a regular width config, while other columns would have a flex config.
								Flex sizing works by dividing the remaining space in the grid among all flex columns
								in proportion to their flex value. For example, suppose the grid has a total width
								of 450px and it has three columns: the first with width: 150; the second with flex:
								1; and third with flex: 2. The first column will be 150px wide, leaving 300px
								remaining. The column with flex: 2 has twice the size with flex: 1. So final sizes
								will be: 150px, 100px, 200px.
							</Tooltip>
						{/snippet}

						<input type="range" step="1" bind:value={value.flex} min={1} max={12} />
						<div class="text-xs">{value.flex}</div>
					</Label>

					<Label label="Hide">
						<Toggle
							on:pointerdown={(e) => {
								e?.stopPropagation()
							}}
							options={{ right: 'Hide' }}
							bind:checked={value.hide}
							size="xs"
						/>
					</Label>

					<Label label="Value formatter">
						{#snippet header()}
							<Tooltip
								documentationLink="https://www.ag-grid.com/javascript-data-grid/value-formatters/"
							>
								Value formatters allow you to format values for display. This is useful when data is
								one type (e.g. numeric) but needs to be converted for human reading (e.g. putting in
								currency symbols and number formatting).
							</Tooltip>
						{/snippet}
						{#snippet action()}
							<Button
								size="xs"
								color="light"
								variant="border"
								on:click={() => {
									// @ts-ignore
									value.valueFormatter = null
									renderCount++
								}}
							>
								Clear
							</Button>
						{/snippet}
					</Label>
					<div>
						{#key renderCount}
							<div class="flex flex-col gap-4">
								<div class="relative">
									{#if !presets.find((preset) => preset.value === value?.valueFormatter)}
										<div
											class="z-50 absolute bg-opacity-50 bg-surface top-0 left-0 bottom-0 right-0"
										></div>
									{/if}
									<div class="text-xs font-semibold">Presets</div>
									<select
										bind:value={value.valueFormatter}
										onchange={() => {
											renderCount++
										}}
										placeholder="Code"
									>
										{#each presets as preset}
											<option value={preset.value}>{preset.label}</option>
										{/each}
									</select>
								</div>

								<SimpleEditor
									extraLib={'declare const value: any'}
									autoHeight
									lang="javascript"
									bind:code={value.valueFormatter}
								/>
								<div class="text-xs text-secondary -mt-4">Use `value` in the formatter</div>
							</div>
						{/key}
					</div>

					<Label label="Sort">
						<select bind:value={value.sort}>
							<option value={null}>None</option>
							<option value="asc">Ascending</option>
							<option value="desc">Descending</option>
						</select>
					</Label>
				</div>
			</Section>
		{/if}
	{/snippet}
</Popover>
