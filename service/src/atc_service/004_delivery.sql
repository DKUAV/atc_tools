ALTER TABLE assets DROP CONSTRAINT assets_kind_check;
ALTER TABLE assets ADD CONSTRAINT assets_kind_check CHECK (kind IN (
    'onnx', 'inspection', 'om', 'simulated_om', 'build_info', 'dataset', 'accuracy', 'profiling',
    'quantized_onnx', 'tensors', 'baseline_om'
));

ALTER TABLE jobs DROP CONSTRAINT jobs_operation_check;
ALTER TABLE jobs ADD CONSTRAINT jobs_operation_check CHECK (operation IN (
    'inspect_onnx', 'compile_om', 'quantize_onnx', 'run_cpu', 'run_npu', 'compare_outputs', 'profile_om', 'retry_step'
));
ALTER TABLE jobs ADD COLUMN lane text NOT NULL DEFAULT 'compile' CHECK (lane IN ('cpu', 'compile', 'device'));
ALTER TABLE jobs ADD COLUMN accuracy_summary jsonb;
UPDATE jobs SET lane = 'cpu' WHERE operation = 'inspect_onnx';
CREATE INDEX queued_jobs_by_lane ON jobs (lane, created_at, id) WHERE state = 'queued';

CREATE TABLE worker_status (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    heartbeat_at timestamptz NOT NULL DEFAULT now(),
    checks jsonb NOT NULL
);
