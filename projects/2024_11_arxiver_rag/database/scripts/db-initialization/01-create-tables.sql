CREATE TABLE paper_information (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    paper_id TEXT NOT NULL, /* arxiver.id */
    published_date TIMESTAMP DEFAULT NOW(), /* arxiver.published_date */
    title TEXT NOT NULL,  /* arxiver.title */
    authors TEXT DEFAULT '' NOT NULL,  /* arxiver.abstract */
    link TEXT DEFAULT '' NOT NULL  /* arxiver.link */
);

CREATE TABLE paper_status(
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    paper_information_id INTEGER REFERENCES paper_information(id) ON DELETE CASCADE,
    file_extension text DEFAULT 'pdf' NOT NULL, /* pdf, docx, .. */
    parse_status text DEFAULT 'PENDING' NOT NULL,
	extract_status text DEFAULT 'PENDING' NOT NULL,
	split_status text DEFAULT 'PENDING' NOT NULL,
	embed_status text DEFAULT 'PENDING' NOT NULL
);