<script setup lang="ts">
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  AlphaPeptInstrumentType,
  FragmentationType,
  SpectrumConfigSource,
  type SpectrumFormModel,
} from "~/utils/types";
import type { KoinaModelConfig } from "~/utils/koina-api";
import ModelSelect from "./ModelSelect.vue";
import PeptideSequenceInput from "./inputs/PeptideSequenceInput.vue";
import PrecursorChargeInput from "./inputs/PrecursorChargeInput.vue";
import CollisionEnergyInput from "./inputs/CollisionEnergyInput.vue";
import InstrumentTypeSelect from "./inputs/InstrumentTypeSelect.vue";
import FragmentationTypeInput from "./inputs/FragmentationTypeInput.vue";

const model = defineModel<SpectrumFormModel>();

if (!model.value) {
  model.value = {
    source: SpectrumConfigSource.PREDICTION,
    model: {
      name: "Prosit_2019_intensity",
      inputs: [],
    },
  };
}

const loading = ref(false);

const modelConfig = ref<KoinaModelConfig>();

watch(
  () => model.value?.model?.name,
  async (newModelName) => {
    if (!newModelName) return;

    loading.value = true;
    try {
      modelConfig.value = await fetchModelTritonConfig(newModelName);
    } finally {
      loading.value = false;
    }
  },
  { immediate: true },
);

function updateModelInput(name: string, value: any): void {
  if (!model.value?.model?.inputs) return;

  const input = model.value.model.inputs.find((input) => input.name === name);

  if (input) {
    input.data = [value];
  } else {
    model.value.model.inputs.push({
      ...EMPTY_KOINA_DATA[name],
      data: [value],
      shape: [1, 1],
    });
  }
}

const peptideSequence = ref<string>("");
const precursorCharge = ref<number>(2);
const collisionEnergy = ref<number>(25);
const instrumentType = ref<AlphaPeptInstrumentType>(AlphaPeptInstrumentType.QE);
const fragmentationType = ref<FragmentationType>(FragmentationType.HCD);

watch(peptideSequence, (newPeptideSequence) => {
  updateModelInput("peptide_sequences", newPeptideSequence);
});

watch(precursorCharge, (newPrecursorCharge) => {
  updateModelInput("precursor_charges", newPrecursorCharge);
});

watch(collisionEnergy, (newCollisionEnergy) => {
  updateModelInput("collision_energies", newCollisionEnergy);
});

watch(instrumentType, (newInstrumentType) => {
  updateModelInput("instrument_types", newInstrumentType);
});

watch(fragmentationType, (newFragmentationType) => {
  updateModelInput("fragmentation_types", newFragmentationType);
});

const modelConfigInputNames = computed(() => {
  return modelConfig.value?.input.map((input) => input.name);
});

watch(
  modelConfigInputNames,
  () => {
    updateModelInput("peptide_sequences", peptideSequence.value);
    updateModelInput("precursor_charges", precursorCharge.value);
    updateModelInput("collision_energies", collisionEnergy.value);
    updateModelInput("instrument_types", instrumentType.value);
    updateModelInput("fragmentation_types", fragmentationType.value);

    if (model.value?.model?.inputs)
      model.value.model.inputs =
        model.value?.model?.inputs.filter((input) =>
          modelConfigInputNames.value?.includes(input.name),
        ) || [];
  },
  { deep: true, immediate: true },
);
</script>

<template>
  <Card>
    <CardHeader>
      <CardTitle>
        <slot name="title" />
      </CardTitle>

      <CardDescription>
        <span class="block mb-2">
          <slot name="description" />
        </span>

        <RadioGroup
          v-model="model!.source"
          :default-value="SpectrumConfigSource.PREDICTION"
        >
          <div class="flex items-center space-x-2">
            <Label
              ><RadioGroupItem :value="SpectrumConfigSource.PREDICTION" /> Koina
              Prediction</Label
            >
          </div>
          <div class="flex items-center space-x-2">
            <Label
              ><RadioGroupItem :value="SpectrumConfigSource.USI" /> USI</Label
            >
          </div>
        </RadioGroup>
      </CardDescription>
    </CardHeader>

    <CardContent class="space-y-2">
      <div
        v-if="model?.source === 'prediction'"
        class="grid grid-cols1 md:grid-cols-2 w-full gap-4"
      >
        <div>
          <Label
            >Prediction Model
            <ModelSelect v-model="model.model!.name" class="mt-1" />
          </Label>
        </div>
        <div class="flex flex-col">
          <PeptideSequenceInput
            class="mb-2"
            v-if="modelConfigInputNames?.includes('peptide_sequences')"
            v-model="peptideSequence"
          />
          <PrecursorChargeInput
            class="mb-2"
            v-if="modelConfigInputNames?.includes('precursor_charges')"
            v-model="precursorCharge"
          />
          <CollisionEnergyInput
            class="mb-2"
            v-if="modelConfigInputNames?.includes('collision_energies')"
            v-model="collisionEnergy"
          />
          <InstrumentTypeSelect
            class="mb-2"
            v-if="modelConfigInputNames?.includes('instrument_types')"
            v-model="instrumentType"
          />
          <FragmentationTypeInput
            class="mb-2"
            v-if="modelConfigInputNames?.includes('fragmentation_types')"
            v-model="fragmentationType"
          />
        </div>
      </div>

      <div v-else-if="model?.source === 'usi'" class="w-full">
        <Label>
          USI
          <Input v-model="model.usi" type="text" placeholder="USI" />
        </Label>
      </div>
    </CardContent>
  </Card>
</template>
