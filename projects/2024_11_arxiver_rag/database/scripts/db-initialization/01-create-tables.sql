CREATE TABLE paper (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    paper_id INTEGER NOT NULL, /* arxiver.id */
    published_date TIMESTAMP DEFAULT NOW(), /* arxiver.published_date */
    title TEXT NOT NULL,  /* arxiver.title */
    authors TEXT DEFAULT '' NOT NULL,  /* arxiver.abstract */
    link TEXT DEFAULT '' NOT NULL  /* arxiver.link */
);