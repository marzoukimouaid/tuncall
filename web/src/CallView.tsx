import { useEffect, useRef, useState } from "react";
import { audioUrl, type Call, getCall, type Segment } from "./api";
import { POLL_MS, StatusBadge } from "./ui";

const time = (s: number) =>
  `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

// Pipeline order; JSONB storage does not keep the key order.
const STAGES = ["diarize", "asr", "nlp"];

const SPEAKER_CLASS = ["border-blue-400", "border-emerald-400", "border-purple-400"];

// D-60a screen 3: transcript by speaker with times, role guesses, per-segment intent /
// entities / sentiment, and audio playback that jumps to a segment.
export default function CallView({ id }: { id: string }) {
  const [call, setCall] = useState<Call | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(0);
  const audio = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    let alive = true;
    const load = () =>
      getCall(id)
        .then((c) => {
          if (!alive) return;
          setCall(c);
          setError(null);
          if (c.status === "queued" || c.status === "processing") {
            timer = setTimeout(load, POLL_MS);
          }
        })
        .catch((e: Error) => {
          if (!alive) return;
          setError(e.message);
          timer = setTimeout(load, POLL_MS); // keep trying: the API may be restarting
        });
    load();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [id]);

  if (!call) return <p className={error ? "text-red-700" : "text-slate-600"}>{error ?? "Loading…"}</p>;

  const record = call.record;
  const speakers = record ? [...new Set(record.segments.map((s) => s.speaker))] : [];
  const roleOf = (spk: string) => record?.segments.find((s) => s.speaker === spk)?.role;

  function play(seg: Segment) {
    if (!audio.current) return;
    audio.current.currentTime = seg.start;
    void audio.current.play();
  }

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-xl font-semibold">{call.filename}</h1>
        <StatusBadge status={call.status} />
      </div>
      {error && <p className="text-sm text-red-700">Cannot reach the API: {error}. Retrying…</p>}
      {call.error && (
        <p className="rounded bg-red-50 p-3 text-sm text-red-800">
          {call.status === "failed" ? "Failed: " : `Attempt ${call.attempts} failed, retrying: `}
          {call.error}
        </p>
      )}
      <audio
        ref={audio}
        controls
        src={audioUrl(call.id)}
        onTimeUpdate={(e) => setNow(e.currentTarget.currentTime)}
        className="w-full"
      />
      {(call.status === "queued" || call.status === "processing") && (
        <p className="text-slate-600">
          {call.status === "queued" ? "Waiting for the worker…" : "Processing…"} This page
          updates by itself.
        </p>
      )}
      {record && (
        <>
          <div className="flex flex-wrap gap-4 text-sm text-slate-600">
            <span>Duration {time(record.duration_s)}</span>
            {speakers.map((spk, i) => (
              <span key={spk} className={`border-l-4 pl-2 ${SPEAKER_CLASS[i % 3]}`}>
                {spk}: {roleOf(spk)}
              </span>
            ))}
            <span>
              Processing{" "}
              {Object.entries(record.meta.processing_ms_by_stage)
                .sort(([a], [b]) => STAGES.indexOf(a) - STAGES.indexOf(b))
                .map(([stage, ms]) => `${stage} ${(ms / 1000).toFixed(1)} s`)
                .join(", ")}
            </span>
          </div>
          <ol className="space-y-2">
            {record.segments.map((seg, i) => {
              const active = now >= seg.start && now < seg.end;
              const color = SPEAKER_CLASS[speakers.indexOf(seg.speaker) % 3];
              return (
                <li
                  key={i}
                  className={`rounded-lg border-l-4 bg-white p-3 shadow-sm ${color} ${
                    active ? "ring-2 ring-blue-300" : ""
                  }`}
                >
                  <div className="mb-1 flex flex-wrap items-center gap-2 text-xs text-slate-500">
                    <button
                      onClick={() => play(seg)}
                      className="font-mono text-blue-700 hover:underline"
                      aria-label={`Play from ${time(seg.start)}`}
                    >
                      ▶ {time(seg.start)}–{time(seg.end)}
                    </button>
                    <span className="font-medium text-slate-700">
                      {seg.speaker} ({seg.role})
                    </span>
                    {seg.intent && <Chip label={`intent: ${seg.intent}`} />}
                    {seg.sentiment && (
                      <Chip
                        label={seg.sentiment}
                        className={
                          seg.sentiment === "positive"
                            ? "bg-green-100 text-green-900"
                            : "bg-red-100 text-red-900"
                        }
                      />
                    )}
                    {seg.entities.map((e, j) => (
                      <Chip key={j} label={`${e.type}: ${e.value}`} />
                    ))}
                  </div>
                  {/* D-10: everything is Arabic script, so the transcript reads right-to-left */}
                  <p dir="rtl" lang="ar" className="text-lg leading-relaxed">
                    {seg.text_raw}
                  </p>
                </li>
              );
            })}
          </ol>
        </>
      )}
    </section>
  );
}

function Chip({ label, className = "bg-slate-100 text-slate-700" }: {
  label: string;
  className?: string;
}) {
  return (
    <span dir="auto" className={`rounded px-1.5 py-0.5 ${className}`}>
      {label}
    </span>
  );
}
