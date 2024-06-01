<template>
  <component
    :is="tag"
    :class="classes"
    :to="props.to"
    :href="props.href"
  >
    <slot />
  </component>
</template>

<script setup lang="ts">
import type { RouteLocationRaw } from 'vue-router'
import { NuxtLink } from '#components'

interface Props {
  color?: 'primary' | 'secondary'
  textColor?: 'white' | 'black'
  tag?: string
  to?: RouteLocationRaw
  href?: string
  outlined?: boolean
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | '2xl' | '3xl'
}

const props = withDefaults(defineProps<Props>(), {
  color: 'primary',
  textColor: undefined,
  outlined: false,
  size: 'xl',
  to: undefined,
  tag: undefined,
  href: undefined,
})

const tag = computed(() => {
  if (props.tag) {
    return props.tag
  }
  else if (props.to) {
    return NuxtLink
  }
  else if (props.href) {
    return 'a'
  }
  else {
    return 'button'
  }
})

const classes = computed(() => {
  const common = ['py-2', 'px-4', 'border-2', 'rounded', `text-${props.size}`]

  const textColor = props.textColor || (props.outlined ? props.color : 'white')
  const hoverTextColor = props.textColor || 'white'

  if (props.outlined) {
    return [
      ...common,
      `border-${props.color}`,
      `text-${textColor}`,
      `hover:bg-${props.color}-dark`,
      `hover:border-${props.color}-dark`,
      `hover:text-${hoverTextColor}`,
    ].join(' ')
  }
  else {
    return [
      ...common,
      `bg-${props.color}`,
      `border-${props.color}`,
      `text-${textColor}`,
      `hover:bg-${props.color}-dark`,
      `hover:border-${props.color}-dark`,
      `hover:text-${hoverTextColor}`,
    ].join(' ')
  }
})
</script>
