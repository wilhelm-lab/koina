<script setup lang="ts">
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { FragmentationType } from "~/utils/types";
import { Label } from "~/components/ui/label";

const model = defineModel<FragmentationType>();
</script>

<template>
  <Label>
    Fragmentation Type
    <Select :default-value="FragmentationType.HCD" v-model="model">
      <SelectTrigger class="mt-1">
        <SelectValue placeholder="Select a fragmentation type..." />
      </SelectTrigger>
      <SelectContent>
        <SelectGroup>
          <SelectLabel>Fragmentation Type</SelectLabel>
          <SelectItem v-for="fragType of FragmentationType" :value="fragType">
            {{ fragType }}
          </SelectItem>
        </SelectGroup>
      </SelectContent>
    </Select>
  </Label>
</template>
