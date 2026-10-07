import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { api, eventsUrl, type ProgressEvent, type Training } from "./api";

export const FINAL_STATUSES = ["done", "failed", "cancelled"] as const;
export const isFinal = (t: Training) => (FINAL_STATUSES as readonly string[]).includes(t.status);

export const useLanguages = () =>
  useQuery({ queryKey: ["languages"], queryFn: api.languages, staleTime: Infinity });

export const useBaseModels = () =>
  useQuery({ queryKey: ["base-models"], queryFn: api.baseModels, staleTime: 60_000 });

export const useWakeWord = (id: string) =>
  useQuery({ queryKey: ["wakeword", id], queryFn: () => api.wakeWord(id) });

export const useTraining = (id: string) =>
  useQuery({ queryKey: ["training", id], queryFn: () => api.training(id) });

/** Server-sent progress of a training; refreshes the training (and its wake word) at the end. */
export function useTrainingEvents(training: Training | undefined) {
  const [events, setEvents] = useState<ProgressEvent[]>([]);
  const client = useQueryClient();
  const active = training !== undefined && !isFinal(training);
  const id = training?.id;
  const wakeWordId = training?.wakeword_id;

  useEffect(() => {
    if (!active || !id) return;
    setEvents([]); // the stream replays every event from the start
    const source = new EventSource(eventsUrl(id));
    source.addEventListener("progress", (e) => {
      const event = JSON.parse((e as MessageEvent).data) as ProgressEvent;
      setEvents((prev) => [...prev, event]);
    });
    const finish = () => {
      source.close(); // the server ends the stream; don't let EventSource reconnect
      client.invalidateQueries({ queryKey: ["training", id] });
      client.invalidateQueries({ queryKey: ["wakeword", wakeWordId] });
    };
    for (const name of [...FINAL_STATUSES, "gone"]) source.addEventListener(name, finish);
    return () => source.close();
  }, [active, id, wakeWordId, client]);

  return events;
}
