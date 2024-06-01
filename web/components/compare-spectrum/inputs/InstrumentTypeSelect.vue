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
import { AlphaPeptInstrumentType } from "~/utils/types";
import { Label } from "~/components/ui/label";

const model = defineModel<AlphaPeptInstrumentType>();
</script>

<template>
  <Label>
    Instrument Type
    <Select>
      <SelectTrigger>
        <SelectValue
          v-model="model"
          placeholder="Select an instrument type..."
        />
      </SelectTrigger>
      <SelectContent>
        <SelectGroup>
          <SelectLabel>Instrument Type</SelectLabel>
          <SelectItem
            v-for="instrument of AlphaPeptInstrumentType"
            :value="instrument"
          >
            {{ instrument }}
          </SelectItem>
        </SelectGroup>
      </SelectContent>
    </Select>
  </Label>
</template>
