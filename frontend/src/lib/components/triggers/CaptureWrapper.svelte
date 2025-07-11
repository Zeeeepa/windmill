<script lang="ts">
	import { workspaceStore } from '$lib/stores'
	import { CaptureService, type CaptureConfig, type CaptureTriggerKind } from '$lib/gen'
	import { onDestroy, untrack } from 'svelte'
	import { isObject, sendUserToast, sleep } from '$lib/utils'
	import RouteCapture from './http/RouteCapture.svelte'
	import type { ConnectionInfo } from '../common/alert/ConnectionIndicator.svelte'
	import type { CaptureInfo } from './CaptureSection.svelte'
	import WebhooksCapture from './webhook/WebhooksCapture.svelte'
	import EmailTriggerCaptures from '../details/EmailTriggerCaptures.svelte'
	import WebsocketCapture from './websocket/WebsocketCapture.svelte'
	import PostgresCapture from './postgres/PostgresCapture.svelte'
	import KafkaCapture from './kafka/KafkaCapture.svelte'
	import NatsCapture from './nats/NatsCapture.svelte'
	import MqttCapture from './mqtt/MqttCapture.svelte'
	import SqsCapture from './sqs/SqsCapture.svelte'
	import GcpCapture from './gcp/GcpCapture.svelte'

	interface Props {
		isFlow: boolean
		path: string
		hasPreprocessor: boolean
		canHavePreprocessor: boolean
		captureType?: CaptureTriggerKind
		data?: any
		connectionInfo?: ConnectionInfo | undefined
		args?: Record<string, any>
		isValid?: boolean
		triggerDeployed?: boolean
	}

	let {
		isFlow,
		path,
		hasPreprocessor,
		canHavePreprocessor,
		captureType = 'webhook',
		data = {},
		connectionInfo = $bindable(undefined),
		args = $bindable({}),
		isValid = false,
		triggerDeployed = false
	}: Props = $props()

	let captureLoading = $state(false)
	let captureActive = $state(false)
	let captureConfigs: {
		[key: string]: CaptureConfig
	} = $state({})
	let ready = $state(false)

	const config: CaptureConfig | undefined = $derived(captureConfigs[captureType])

	export async function setConfig(): Promise<boolean> {
		try {
			await CaptureService.setCaptureConfig({
				requestBody: {
					trigger_kind: captureType,
					path,
					is_flow: isFlow,
					trigger_config: args && Object.keys(args).length > 0 ? args : undefined
				},
				workspace: $workspaceStore!
			})
			return true
		} catch (error) {
			sendUserToast(error.body, true)
			return false
		}
	}

	function isStreamingCapture() {
		if (captureType === 'gcp' && args.delivery_type === 'push') {
			return false
		}
		return ['mqtt', 'sqs', 'websocket', 'postgres', 'kafka', 'nats', 'gcp'].includes(captureType)
	}

	async function getCaptureConfigs() {
		const captureConfigsList = await CaptureService.getCaptureConfigs({
			workspace: $workspaceStore!,
			runnableKind: isFlow ? 'flow' : 'script',
			path
		})

		captureConfigs = captureConfigsList.reduce((acc, c) => {
			acc[c.trigger_kind] = c
			return acc
		}, {})

		if (isStreamingCapture() && captureActive) {
			const config = captureConfigs[captureType]
			if (config && config.error) {
				const serverEnabled = getServerEnabled(config)
				if (!serverEnabled) {
					sendUserToast('Capture was stopped because of error: ' + config.error, true)
					captureActive = false
				}
			}
		}
		return captureConfigs
	}
	getCaptureConfigs().then((captureConfigs) => setDefaultArgs(captureConfigs))

	async function capture() {
		let i = 0
		captureActive = true
		while (captureActive) {
			if (i % 3 === 0) {
				await CaptureService.pingCaptureConfig({
					workspace: $workspaceStore!,
					triggerKind: captureType,
					runnableKind: isFlow ? 'flow' : 'script',
					path
				})
				await getCaptureConfigs()
			}
			i++
			await sleep(1000)
		}
	}

	function setDefaultArgs(captureConfigs: { [key: string]: CaptureConfig }) {
		if (captureType in captureConfigs) {
			const triggerConfig = captureConfigs[captureType].trigger_config
			args = isObject(triggerConfig) ? triggerConfig : {}
		} else {
			args = {}
		}
		ready = true
	}

	onDestroy(() => {
		captureActive = false
	})

	function getServerEnabled(config: CaptureConfig) {
		return (
			!!config.last_server_ping &&
			new Date(config.last_server_ping).getTime() > new Date().getTime() - 15 * 1000
		)
	}

	export async function handleCapture(e: CustomEvent<{ disableOnly?: boolean }>) {
		if (captureActive || e.detail.disableOnly) {
			captureActive = false
		} else {
			try {
				captureLoading = true
				const configSet = await setConfig()

				if (configSet) {
					capture()
				}
			} finally {
				captureLoading = false
			}
		}
	}

	function updateConnectionInfo(config: CaptureConfig | undefined, captureActive: boolean) {
		if (isStreamingCapture() && config && captureActive) {
			const serverEnabled = getServerEnabled(config)
			const connected = serverEnabled && !config.error
			const message = connected
				? `Connected`
				: `Not connected${config.error ? ': ' + config.error : ''}`
			connectionInfo = {
				connected,
				message
			}
		} else {
			connectionInfo = undefined
		}
	}
	$effect(() => {
		const args = [config, captureActive] as const
		untrack(() => updateConnectionInfo(...args))
	})

	let captureInfo: CaptureInfo = $derived({
		active: captureActive,
		hasPreprocessor,
		canHavePreprocessor,
		isFlow,
		path,
		connectionInfo
	})

	$effect(() => {
		args && (captureActive = false)
	})
</script>

{#key ready}
	<div class="flex flex-col gap-4 w-full h-full">
		{#if captureType === 'websocket'}
			<WebsocketCapture
				{isValid}
				{captureInfo}
				{captureLoading}
				on:applyArgs
				on:updateSchema
				on:addPreprocessor
				on:captureToggle={handleCapture}
				on:testWithArgs
			/>
		{:else if captureType === 'postgres'}
			<PostgresCapture
				{captureInfo}
				{captureLoading}
				{isValid}
				{hasPreprocessor}
				{isFlow}
				{triggerDeployed}
				on:applyArgs
				on:updateSchema
				on:addPreprocessor
				on:captureToggle={handleCapture}
				on:testWithArgs
			/>
		{:else if captureType === 'webhook'}
			<WebhooksCapture
				{hasPreprocessor}
				{isFlow}
				{path}
				runnableArgs={data?.args}
				{captureInfo}
				{captureLoading}
				on:applyArgs
				on:updateSchema
				on:addPreprocessor
				on:captureToggle={handleCapture}
				on:testWithArgs
			/>
		{:else if captureType === 'http'}
			<RouteCapture
				runnableArgs={data?.args}
				route_path={args.route_path}
				http_method={args.http_method}
				{isValid}
				{captureInfo}
				{hasPreprocessor}
				{isFlow}
				{captureLoading}
				on:applyArgs
				on:updateSchema
				on:addPreprocessor
				on:captureToggle={handleCapture}
				on:testWithArgs
			/>
		{:else if captureType === 'email'}
			<EmailTriggerCaptures
				{path}
				{isFlow}
				emailDomain={data?.emailDomain}
				{captureInfo}
				{hasPreprocessor}
				{captureLoading}
				on:applyArgs
				on:updateSchema
				on:addPreprocessor
				on:captureToggle={handleCapture}
				on:testWithArgs
			/>
		{:else if captureType === 'kafka'}
			<KafkaCapture
				{isValid}
				{captureInfo}
				{hasPreprocessor}
				{isFlow}
				{captureLoading}
				{triggerDeployed}
				on:applyArgs
				on:updateSchema
				on:addPreprocessor
				on:captureToggle={handleCapture}
				on:testWithArgs
			/>
		{:else if captureType === 'nats'}
			<NatsCapture
				{isValid}
				{captureInfo}
				{hasPreprocessor}
				{isFlow}
				{captureLoading}
				{triggerDeployed}
				on:applyArgs
				on:updateSchema
				on:addPreprocessor
				on:captureToggle={handleCapture}
				on:testWithArgs
			/>
		{:else if captureType === 'mqtt'}
			<MqttCapture
				{isValid}
				{captureInfo}
				{hasPreprocessor}
				{isFlow}
				{captureLoading}
				{triggerDeployed}
				on:applyArgs
				on:updateSchema
				on:addPreprocessor
				on:captureToggle={handleCapture}
				on:testWithArgs
			/>
		{:else if captureType === 'sqs'}
			<SqsCapture
				{isValid}
				{captureInfo}
				{hasPreprocessor}
				{isFlow}
				{captureLoading}
				{triggerDeployed}
				on:applyArgs
				on:updateSchema
				on:addPreprocessor
				on:captureToggle={handleCapture}
				on:testWithArgs
			/>
		{:else if captureType === 'gcp'}
			<GcpCapture
				{isValid}
				{captureInfo}
				{hasPreprocessor}
				{isFlow}
				{triggerDeployed}
				deliveryType={args.delivery_type}
				{captureLoading}
				on:applyArgs
				on:updateSchema
				on:addPreprocessor
				on:captureToggle={handleCapture}
				on:testWithArgs
			/>
		{/if}
	</div>
{/key}
