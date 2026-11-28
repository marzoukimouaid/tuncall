import { useCallback, useEffect, useState } from "react";
import { type Call, deleteCall, listCalls, uploadCall } from "./api";
import CallView from "./CallView";
import { POLL_MS, StatusBadge } from "./ui";

// Hash routes keep the dashboard a plain static bundle: #/ (list), #/upload, #/calls/<id>.
function useRoute(): string {
  const [route, setRoute] = useState(() => location.hash.slice(1) || "/");
  useEffect(() => {
    const onHash = () => setRoute(location.hash.slice(1) || "/");
    addEventListener("hashchange", onHash);
    return () => removeEventListener("hashchange", onHash);
  }, []);
  return route;
}

export const go = (route: string) => {
  location.hash = route;
};

export default function App() {
  const route = useRoute();
  const callId = route.match(/^\/calls\/([0-9a-f-]+)$/)?.[1];
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <nav className="mx-auto flex max-w-5xl items-center gap-6 px-4 py-3">
          <a href="#/" className="text-lg font-semibold">
            TunCall
          </a>
          <a href="#/" className={navClass(route === "/" || !!callId)}>
            Calls
          </a>
          <a href="#/upload" className={navClass(route === "/upload")}>
            Upload
          </a>
        </nav>
      </header>
      <main className="mx-auto max-w-5xl px-4 py-6">
        {callId ? <CallView id={callId} /> : route === "/upload" ? <Upload /> : <CallList />}
      </main>
    </div>
  );
}

const navClass = (active: boolean) =>
  active ? "font-medium text-blue-700" : "text-slate-600 hover:text-slate-900";

function Upload() {
  const [file, setFile] = useState<File | null>(null);
  const [speakers, setSpeakers] = useState(2);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      go(`/calls/${await uploadCall(file, speakers)}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="max-w-lg space-y-4 rounded-lg bg-white p-6 shadow-sm">
      <h1 className="text-xl font-semibold">Upload a call</h1>
      <p className="text-sm text-slate-600">
        WAV, FLAC, OGG, MP3 or M4A, up to 30 minutes. The call is queued and processed in the
        background.
      </p>
      <label className="block">
        <span className="text-sm font-medium">Audio file</span>
        <input
          type="file"
          accept="audio/*,.m4a,.aac"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="mt-1 block w-full text-sm"
        />
      </label>
      <label className="block">
        <span className="text-sm font-medium">Number of speakers</span>
        <input
          type="number"
          min={1}
          max={10}
          value={speakers}
          onChange={(e) => setSpeakers(Number(e.target.value))}
          className="mt-1 block w-24 rounded border border-slate-300 px-2 py-1"
        />
      </label>
      {error && <p className="text-sm text-red-700">{error}</p>}
      <button
        type="submit"
        disabled={!file || busy || !(speakers >= 1 && speakers <= 10)}
        className="rounded bg-blue-700 px-4 py-2 text-white disabled:opacity-50"
      >
        {busy ? "Uploading…" : "Upload"}
      </button>
    </form>
  );
}

function CallList() {
  const [calls, setCalls] = useState<Call[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    listCalls()
      .then((c) => {
        setCalls(c);
        setError(null);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, POLL_MS);
    return () => clearInterval(timer);
  }, [refresh]);

  async function remove(call: Call) {
    if (!confirm(`Delete ${call.filename} and its audio?`)) return;
    await deleteCall(call.id).catch((e: Error) => setError(e.message));
    refresh();
  }

  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Calls</h1>
        <a href="#/upload" className="rounded bg-blue-700 px-3 py-1.5 text-sm text-white">
          Upload
        </a>
      </div>
      {error && <p className="text-sm text-red-700">{error}</p>}
      {calls?.length === 0 && <p className="text-slate-600">No calls yet.</p>}
      {calls && calls.length > 0 && (
        <table className="w-full overflow-hidden rounded-lg bg-white text-sm shadow-sm">
          <thead className="bg-slate-100 text-left text-slate-600">
            <tr>
              <th className="px-3 py-2">File</th>
              <th className="px-3 py-2">Status</th>
              <th className="px-3 py-2">Uploaded</th>
              <th className="px-3 py-2" />
            </tr>
          </thead>
          <tbody>
            {calls.map((c) => (
              <tr key={c.id} className="border-t border-slate-100">
                <td className="px-3 py-2">
                  <a href={`#/calls/${c.id}`} className="text-blue-700 hover:underline">
                    {c.filename}
                  </a>
                </td>
                <td className="px-3 py-2">
                  <StatusBadge status={c.status} />
                  {c.error && (
                    <span className="ml-2 text-xs text-red-700" title={c.error}>
                      {c.status === "failed" ? "error" : `retrying (attempt ${c.attempts})`}
                    </span>
                  )}
                </td>
                <td className="px-3 py-2 text-slate-600">
                  {new Date(c.created_at).toLocaleString()}
                </td>
                <td className="px-3 py-2 text-right">
                  <button
                    onClick={() => remove(c)}
                    className="text-slate-500 hover:text-red-700"
                    aria-label={`Delete ${c.filename}`}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
