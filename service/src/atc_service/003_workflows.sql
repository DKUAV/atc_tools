ALTER TABLE assets DROP CONSTRAINT assets_kind_check;
ALTER TABLE assets ADD CONSTRAINT assets_kind_check CHECK (kind IN (
    'onnx', 'inspection', 'om', 'simulated_om', 'build_info', 'dataset', 'accuracy', 'profiling'
));

ALTER TABLE jobs ADD COLUMN workflow jsonb;
ALTER TABLE jobs ADD COLUMN stages jsonb NOT NULL DEFAULT '{}';
