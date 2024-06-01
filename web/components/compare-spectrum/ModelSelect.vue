<script setup lang="ts">
import { ref } from "vue";
import { Check, ChevronsUpDown } from "lucide-vue-next";
import { AVAILABLE_MODELS } from "@/utils/constants";

import { cn } from "@/utils/shadcn";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

import type { HTMLAttributes } from "vue";

const props = defineProps<{ class?: HTMLAttributes["class"] }>();

const open = ref(false);
const model = defineModel<string>();
</script>

<template>
  <Popover v-model:open="open">
    <PopoverTrigger as-child>
      <Button
        variant="outline"
        role="combobox"
        :aria-expanded="open"
        :class="cn('w-full justify-between', props.class)"
      >
        {{ model || "Select model..." }}
        <ChevronsUpDown class="ml-2 h-4 w-4 shrink-0 opacity-50" />
      </Button>
    </PopoverTrigger>

    <PopoverContent class="p-0">
      <Command>
        <CommandInput class="h-9" placeholder="Search model..." />
        <CommandEmpty>No models found.</CommandEmpty>
        <CommandList>
          <CommandGroup>
            <CommandItem
              v-for="predModel in AVAILABLE_MODELS"
              :key="predModel"
              :value="predModel"
              @select="
                (ev) => {
                  if (typeof ev.detail.value === 'string') {
                    model = ev.detail.value;
                  }
                  open = false;
                }
              "
            >
              {{ predModel }}
              <Check
                :class="
                  cn(
                    'ml-auto h-4 w-4',
                    model === predModel ? 'opacity-100' : 'opacity-0',
                  )
                "
              />
            </CommandItem>
          </CommandGroup>
        </CommandList>
      </Command>
    </PopoverContent>
  </Popover>
</template>
