<script setup lang="ts">
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";

const model = defineModel<number>();
</script>

<template>
  <Label>
    Precursor Charge
    <Input v-model="model" type="number" placeholder="Precursor charge..." />
  </Label>
</template>
