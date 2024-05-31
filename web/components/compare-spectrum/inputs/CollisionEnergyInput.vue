<script setup lang="ts">
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";

const model = defineModel<number>();
</script>

<template>
  <Label>
    Collision Energy
    <Input v-model="model" type="number" placeholder="Collision energy..." />
  </Label>
</template>
