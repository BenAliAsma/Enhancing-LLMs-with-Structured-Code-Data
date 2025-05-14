# Database call in loop
Database calls in loops are slower than running a single query and consume more resources. This can lead to denial of service attacks if the loop bounds can be controlled by an attacker.


## Recommendation
Ensure that where possible, database queries are not run in a loop, instead running a single query to get all relevant data.


## Example
In the example below, users in a database are queried one by one in a loop:


```go
package main

import "gorm.io/gorm"

func getUsers(db *gorm.DB, names []string) []User {
	res := make([]User, 0, len(names))
	for _, name := range names {
		var user User
		db.Where("name = ?", name).First(&user)
		res = append(res, user)
	}
	return res
}

```
This is corrected by running a single query that selects all of the users at once:


```go
package main

import "gorm.io/gorm"

func getUsersGood(db *gorm.DB, names []string) []User {
	res := make([]User, 0, len(names))
	db.Where("name IN ?", names).Find(&res)
	return res
}

```

## References
