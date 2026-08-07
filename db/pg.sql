CREATE TABLE IF NOT EXISTS persons (
    person_id VARCHAR(64) PRIMARY KEY,
    gender VARCHAR(16),
    age_group VARCHAR(32),
    metadata JSONB DEFAULT '{}'::jsonb,
    inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE IF NOT EXISTS stores (
    store_id VARCHAR(64) PRIMARY KEY,
    store VARCHAR(255) NOT NULL, 
    camera_url VARCHAR(255) NOT NULL, 
    region VARCHAR(255) NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
); 

CREATE TABLE IF NOT EXISTS entry_exit_logs (
    id SERIAL PRIMARY KEY,
    person_id VARCHAR(64) REFERENCES persons(person_id),
    store_id VARCHAR(64) REFERENCES stores(store_id),
    type VARCHAR(8) CHECK (type IN ('entry', 'exit')),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
