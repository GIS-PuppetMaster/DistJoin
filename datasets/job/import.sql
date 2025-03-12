CREATE TABLE aka_name (

    id integer NOT NULL PRIMARY KEY,

    person_id integer NOT NULL,

    name character varying,

    imdb_index character varying(3),

    name_pcode_cf character varying(11),

    name_pcode_nf character varying(11),

    surname_pcode character varying(11),

    md5sum character varying(65)

);



CREATE TABLE aka_title (

    id integer NOT NULL PRIMARY KEY,

    movie_id integer NOT NULL,

    title character varying,

    imdb_index character varying(4),

    kind_id integer NOT NULL,

    production_year integer,

    phonetic_code character varying(5),

    episode_of_id integer,

    season_nr integer,

    episode_nr integer,

    note character varying(72),

    md5sum character varying(32)

);



CREATE TABLE cast_info (

    id integer NOT NULL PRIMARY KEY,

    person_id integer NOT NULL,

    movie_id integer NOT NULL,

    person_role_id integer,

    note character varying,

    nr_order integer,

    role_id integer NOT NULL

);



CREATE TABLE char_name (

    id integer NOT NULL PRIMARY KEY,

    name character varying NOT NULL,

    imdb_index character varying(2),

    imdb_id integer,

    name_pcode_nf character varying(5),

    surname_pcode character varying(5),

    md5sum character varying(32)

);



CREATE TABLE comp_cast_type (

    id integer NOT NULL PRIMARY KEY,

    kind character varying(32) NOT NULL

);



CREATE TABLE company_name (

    id integer NOT NULL PRIMARY KEY,

    name character varying NOT NULL,

    country_code character varying(6),

    imdb_id integer,

    name_pcode_nf character varying(5),

    name_pcode_sf character varying(5),

    md5sum character varying(32)

);



CREATE TABLE company_type (

    id integer NOT NULL PRIMARY KEY,

    kind character varying(32)

);



CREATE TABLE complete_cast (

    id integer NOT NULL PRIMARY KEY,

    movie_id integer,

    subject_id integer NOT NULL,

    status_id integer NOT NULL

);



CREATE TABLE info_type (

    id integer NOT NULL PRIMARY KEY,

    info character varying(32) NOT NULL

);



CREATE TABLE keyword (

    id integer NOT NULL PRIMARY KEY,

    keyword character varying NOT NULL,

    phonetic_code character varying(5)

);



CREATE TABLE kind_type (

    id integer NOT NULL PRIMARY KEY,

    kind character varying(15)

);



CREATE TABLE link_type (

    id integer NOT NULL PRIMARY KEY,

    link character varying(32) NOT NULL

);



CREATE TABLE movie_companies (

    id integer NOT NULL PRIMARY KEY,

    movie_id integer NOT NULL,

    company_id integer NOT NULL,

    company_type_id integer NOT NULL,

    note character varying

);



CREATE TABLE movie_info_idx (

    id integer NOT NULL PRIMARY KEY,

    movie_id integer NOT NULL,

    info_type_id integer NOT NULL,

    info character varying NOT NULL,

    note character varying(1)

);



CREATE TABLE movie_keyword (

    id integer NOT NULL PRIMARY KEY,

    movie_id integer NOT NULL,

    keyword_id integer NOT NULL

);



CREATE TABLE movie_link (

    id integer NOT NULL PRIMARY KEY,

    movie_id integer NOT NULL,

    linked_movie_id integer NOT NULL,

    link_type_id integer NOT NULL

);



CREATE TABLE name (

    id integer NOT NULL PRIMARY KEY,

    name character varying NOT NULL,

    imdb_index character varying(9),

    imdb_id integer,

    gender character varying(1),

    name_pcode_cf character varying(5),

    name_pcode_nf character varying(5),

    surname_pcode character varying(5),

    md5sum character varying(32)

);



CREATE TABLE role_type (

    id integer NOT NULL PRIMARY KEY,

    role character varying(32) NOT NULL

);



CREATE TABLE title (

    id integer NOT NULL PRIMARY KEY,

    title character varying NOT NULL,

    imdb_index character varying(5),

    kind_id integer NOT NULL,

    production_year integer,

    imdb_id integer,

    phonetic_code character varying(5),

    episode_of_id integer,

    season_nr integer,

    episode_nr integer,

    series_years character varying(49),

    md5sum character varying(32)

);



CREATE TABLE movie_info (

    id integer NOT NULL PRIMARY KEY,

    movie_id integer NOT NULL,

    info_type_id integer NOT NULL,

    info character varying NOT NULL,

    note character varying

);



CREATE TABLE person_info (

    id integer NOT NULL PRIMARY KEY,

    person_id integer NOT NULL,

    info_type_id integer NOT NULL,

    info character varying NOT NULL,

    note character varying

);

COPY aka_name FROM '/home/zkx/AQP/datasets/job/aka_name.csv' WITH (FORMAT CSV, DELIMITER ',', QUOTE '"', ESCAPE '\', HEADER);
COPY char_name FROM '/home/zkx/AQP/datasets/job/char_name.csv' WITH (FORMAT CSV, DELIMITER ',', QUOTE '"', ESCAPE '\', HEADER);
COPY movie_keyword FROM '/home/zkx/AQP/datasets/job/movie_keyword.csv' WITH (FORMAT CSV, DELIMITER ',', QUOTE '"', ESCAPE '\', HEADER);
COPY movie_info_idx FROM '/home/zkx/AQP/datasets/job/movie_info_idx.csv' WITH (FORMAT CSV, DELIMITER ',', QUOTE '"', ESCAPE '\', HEADER);
COPY movie_info FROM '/home/zkx/AQP/datasets/job/movie_info.csv' WITH (FORMAT CSV, DELIMITER ',', QUOTE '"', ESCAPE '\', HEADER);
COPY movie_companies FROM '/home/zkx/AQP/datasets/job/movie_companies.csv' WITH (FORMAT CSV, DELIMITER ',', QUOTE '"', ESCAPE '\', HEADER);
COPY person_info FROM '/home/zkx/AQP/datasets/job/person_info.csv' WITH (FORMAT CSV, DELIMITER ',', QUOTE '"', ESCAPE '\', HEADER);
COPY aka_title FROM '/home/zkx/AQP/datasets/job/aka_title.csv' WITH (FORMAT CSV, DELIMITER ',', QUOTE '"', ESCAPE '\', HEADER);
COPY info_type FROM '/home/zkx/AQP/datasets/job/info_type.csv' WITH (FORMAT CSV, DELIMITER ',', QUOTE '"', ESCAPE '\', HEADER);
COPY company_name FROM '/home/zkx/AQP/datasets/job/company_name.csv' WITH (FORMAT CSV, DELIMITER ',', QUOTE '"', ESCAPE '\', HEADER);
COPY name FROM '/home/zkx/AQP/datasets/job/name.csv' WITH (FORMAT CSV, DELIMITER ',', QUOTE '"', ESCAPE '\', HEADER);
COPY link_type FROM '/home/zkx/AQP/datasets/job/link_type.csv' WITH (FORMAT CSV, DELIMITER ',', QUOTE '"', ESCAPE '\', HEADER);
COPY comp_cast_type FROM '/home/zkx/AQP/datasets/job/comp_cast_type.csv' WITH (FORMAT CSV, DELIMITER ',', QUOTE '"', ESCAPE '\', HEADER);
COPY title FROM '/home/zkx/AQP/datasets/job/title.csv' WITH (FORMAT CSV, DELIMITER ',', QUOTE '"', ESCAPE '\', HEADER);
COPY complete_cast FROM '/home/zkx/AQP/datasets/job/complete_cast.csv' WITH (FORMAT CSV, DELIMITER ',', QUOTE '"', ESCAPE '\', HEADER);
COPY company_type FROM '/home/zkx/AQP/datasets/job/company_type.csv' WITH (FORMAT CSV, DELIMITER ',', QUOTE '"', ESCAPE '\', HEADER);
COPY cast_info FROM '/home/zkx/AQP/datasets/job/cast_info.csv' WITH (FORMAT CSV, DELIMITER ',', QUOTE '"', ESCAPE '\', HEADER);
COPY movie_link FROM '/home/zkx/AQP/datasets/job/movie_link.csv' WITH (FORMAT CSV, DELIMITER ',', QUOTE '"', ESCAPE '\', HEADER);
COPY role_type FROM '/home/zkx/AQP/datasets/job/role_type.csv' WITH (FORMAT CSV, DELIMITER ',', QUOTE '"', ESCAPE '\', HEADER);
COPY kind_type FROM '/home/zkx/AQP/datasets/job/kind_type.csv' WITH (FORMAT CSV, DELIMITER ',', QUOTE '"', ESCAPE '\', HEADER);
COPY keyword FROM '/home/zkx/AQP/datasets/job/keyword.csv' WITH (FORMAT CSV, DELIMITER ',', QUOTE '"', ESCAPE '\', HEADER);