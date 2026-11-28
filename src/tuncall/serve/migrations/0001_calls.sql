-- D-55c: a Call has a status, an attempt count and at most one CallRecord (JSONB, D-53b).
-- D-51a: this table is also the job queue; D-55d: claimed_at is the lease start.
CREATE TABLE calls (
    id           uuid PRIMARY KEY,
    filename     text NOT NULL,
    audio_path   text NOT NULL,
    num_speakers integer CHECK (num_speakers > 0),
    status       text NOT NULL DEFAULT 'queued'
                 CHECK (status IN ('queued', 'processing', 'done', 'failed')),
    attempts     integer NOT NULL DEFAULT 0,
    claimed_at   timestamptz,
    finished_at  timestamptz,
    error        text,
    record       jsonb,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX calls_created_at ON calls (created_at);
CREATE INDEX calls_pending ON calls (created_at) WHERE status IN ('queued', 'processing');
