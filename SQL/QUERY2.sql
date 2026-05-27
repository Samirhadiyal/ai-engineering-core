USE instagram;

SELECT COUNT(id)
FROM user1
GROUP BY age;



SELECT age, max(followers) 
FROM user1
GROUP BY age;
	

SELECT age, max(followers) 
FROM user1
GROUP BY age
HAVING max(followers) > 100
ORDER BY age DESC;
