BEGIN;
-- 导入 aka_name 表
/*
COPY aka_name (id, person_id, name, imdb_index, name_pcode_cf, name_pcode_nf, surname_pcode, md5sum)
FROM '/mnt/d/PycharmProjects/AQP/datasets/job/aka_name_previous.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '\');

-- 导入 aka_title 表
COPY aka_title (id, movie_id, title, imdb_index, kind_id, production_year, phonetic_code, episode_of_id, season_nr, episode_nr, note, md5sum)
FROM '/mnt/d/PycharmProjects/AQP/datasets/job/aka_title_previous.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '\');
*/
-- 导入 cast_info 表
COPY cast_info (id, person_id, movie_id, person_role_id, note, nr_order, role_id)
FROM '/mnt/d/PycharmProjects/AQP/datasets/job/cast_info_previous.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '\');
/*
-- 导入 char_name 表
COPY char_name (id, name, imdb_index, imdb_id, name_pcode_nf, surname_pcode, md5sum)
FROM '/mnt/d/PycharmProjects/AQP/datasets/job/char_name_previous.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '\');

-- 导入 comp_cast_type 表
COPY comp_cast_type (id, kind)
FROM '/mnt/d/PycharmProjects/AQP/datasets/job/comp_cast_type_previous.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '\');

-- 导入 company_name 表
COPY company_name (id, name, country_code, imdb_id, name_pcode_nf, name_pcode_sf, md5sum)
FROM '/mnt/d/PycharmProjects/AQP/datasets/job/company_name_previous.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '\');

-- 导入 company_type 表
COPY company_type (id, kind)
FROM '/mnt/d/PycharmProjects/AQP/datasets/job/company_type_previous.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '\');

-- 导入 complete_cast 表
COPY complete_cast (id, movie_id, subject_id, status_id)
FROM '/mnt/d/PycharmProjects/AQP/datasets/job/complete_cast_previous.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '\');

-- 导入 info_type 表
COPY info_type (id, info)
FROM '/mnt/d/PycharmProjects/AQP/datasets/job/info_type_previous.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '\');

-- 导入 keyword 表
COPY keyword (id, keyword, phonetic_code)
FROM '/mnt/d/PycharmProjects/AQP/datasets/job/keyword_previous.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '\');

-- 导入 kind_type 表
COPY kind_type (id, kind)
FROM '/mnt/d/PycharmProjects/AQP/datasets/job/kind_type_previous.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '\');

-- 导入 link_type 表
COPY link_type (id, link)
FROM '/mnt/d/PycharmProjects/AQP/datasets/job/link_type_previous.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '\');
*/
-- 导入 movie_companies 表
COPY movie_companies (id, movie_id, company_id, company_type_id, note)
FROM '/mnt/d/PycharmProjects/AQP/datasets/job/movie_companies_previous.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '\');

-- 导入 movie_info_idx 表
COPY movie_info_idx (id, movie_id, info_type_id, info, note)
FROM '/mnt/d/PycharmProjects/AQP/datasets/job/movie_info_idx_previous.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '\');

-- 导入 movie_keyword 表
COPY movie_keyword (id, movie_id, keyword_id)
FROM '/mnt/d/PycharmProjects/AQP/datasets/job/movie_keyword_previous.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '\');
/*
-- 导入 movie_link 表
COPY movie_link (id, movie_id, linked_movie_id, link_type_id)
FROM '/mnt/d/PycharmProjects/AQP/datasets/job/movie_link_previous.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '\');

-- 导入 name 表
COPY name (id, name, imdb_index, imdb_id, gender, name_pcode_cf, name_pcode_nf, surname_pcode, md5sum)
FROM '/mnt/d/PycharmProjects/AQP/datasets/job/name_previous.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '\');

-- 导入 role_type 表
COPY role_type (id, role)
FROM '/mnt/d/PycharmProjects/AQP/datasets/job/role_type_previous.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '\');
*/
-- 导入 title 表
COPY title (id, title, imdb_index, kind_id, production_year, imdb_id, phonetic_code, episode_of_id, season_nr, episode_nr, series_years, md5sum)
FROM '/mnt/d/PycharmProjects/AQP/datasets/job/title_previous.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '\');

-- 导入 movie_info 表
COPY movie_info (id, movie_id, info_type_id, info, note)
FROM '/mnt/d/PycharmProjects/AQP/datasets/job/movie_info_previous.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '\');
/*
-- 导入 person_info 表
COPY person_info (id, person_id, info_type_id, info, note)
FROM '/mnt/d/PycharmProjects/AQP/datasets/job/person_info_previous.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '\');
*/
COMMIT;

