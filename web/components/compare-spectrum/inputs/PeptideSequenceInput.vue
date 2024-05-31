<script setup lang="ts">
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";

const model = defineModel<string>();
</script>

<template>
  <Label>
    Peptide Sequence
    <Input v-model="model" type="text" placeholder="Peptide sequence..." />
  </Label>
</template>
