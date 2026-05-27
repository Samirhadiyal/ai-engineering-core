USE instagram;

CREATE TABLE user (
    id INT NOT NULL,
    user_id VARCHAR(30),
    email VARCHAR(30) UNIQUE,
    followers INT DEFAULT 0,
    following_ INT DEFAULT 0,
    PRIMARY KEY (id)
);


CREATE TABLE post (
    post_id INT NOT NULL,
    content VARCHAR(60),
    user_id INT,
    PRIMARY KEY (post_id),
    FOREIGN KEY (user_id) REFERENCES user(id)
);

SELECT id, followers FROM user;

INSERT INTO user 
(id,user_id,email,followers,following_)
VALUE 
(2, 'u12346', 'john@gmail.com', 120, 90),
(3, 'u12347', 'anna@yahoo.com', 540, 300),
(4, 'u12348', 'mike@outlook.com', 75, 110),
(5, 'u12349', 'sara@gmail.com', 980, 450),
(6, 'u12350', 'alex@protonmail.com', 30, 25);

SELECT * FROM user;

SELECT DISTINCT email FROM user;




