CREATE TABLE pictures (
    id SERIAL PRIMARY KEY,
    path VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    segments jsonb DEFAULT NULL
);
