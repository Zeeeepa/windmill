<script lang="ts">
	import { useSvelteFlow } from '@xyflow/svelte'
	import { untrack } from 'svelte'

	let { width } = $props()
	const { setViewport, getViewport } = useSvelteFlow()

	$effect(() => {
		;[width]
		untrack(() => onWidthChange(width))
	})
	let lastWidth: number | undefined = undefined

	function onWidthChange(width: number) {
		if (lastWidth === width) return
		const viewport = getViewport()
		let diff = width - (lastWidth ?? 0)
		lastWidth = width
		setViewport({
			...viewport,
			x: viewport.x + diff / 2
		})
	}
</script>
