# Unknown query
Web Cache Deception is a security vulnerability where an attacker tricks a web server into caching sensitive information and then accesses that cached data.

This attack exploits certain behaviors in caching mechanisms by requesting URLs that trick the server into thinking that a non-cachable page is cachable. If a user then accesses sensitive information on these pages, it could be cached and later retrieved by the attacker.


## Recommendation
To prevent Web Cache Deception attacks, web applications should clearly define cacheable and non-cacheable resources. Implementing strict cache controls and validating requested URLs can mitigate the risk of sensitive data being cached.


## Example
Vulnerable code example: A web server is configured to cache all responses ending in '.css'. An attacker requests 'profile.css', and the server processes 'profile', a sensitive page, and caches it.


```go
package bad

import (
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os/exec"
	"strings"
	"sync"
)

var sessionMap = make(map[string]string)

var (
	templateCache = make(map[string]*template.Template)
	mutex         = &sync.Mutex{}
)

type Lists struct {
	Uid       string
	UserName  string
	UserLists []string
	ReadFile  func(filename string) string
}

func parseTemplateFile(templateName string, tmplFile string) (*template.Template, error) {
	mutex.Lock()
	defer mutex.Unlock()

	// Check if the template is already cached
	if cachedTemplate, ok := templateCache[templateName]; ok {
		fmt.Println("cached")
		return cachedTemplate, nil
	}

	// Parse and store the template in the cache
	parsedTemplate, _ := template.ParseFiles(tmplFile)
	fmt.Println("not cached")

	templateCache[templateName] = parsedTemplate
	return parsedTemplate, nil
}

func ShowAdminPageCache(w http.ResponseWriter, r *http.Request) {

	if r.Method == "GET" {
		fmt.Println("cache called")
		sessionMap[r.RequestURI] = "admin"

		// Check if a session value exists
		if _, ok := sessionMap[r.RequestURI]; ok {
			cmd := "mysql -h mysql -u root -prootwolf -e 'select id,name,mail,age,created_at,updated_at from vulnapp.user where name not in (\"" + "admin" + "\");'"

			// mysql -h mysql -u root -prootwolf -e 'select id,name,mail,age,created_at,updated_at from vulnapp.user where name not in ("test");--';echo");'
			fmt.Println(cmd)

			res, err := exec.Command("sh", "-c", cmd).Output()
			if err != nil {
				fmt.Println("err : ", err)
			}

			splitedRes := strings.Split(string(res), "\n")

			p := Lists{Uid: "1", UserName: "admin", UserLists: splitedRes}

			parsedTemplate, _ := parseTemplateFile("page", "./views/admin/userlists.gtpl")
			w.Header().Set("Cache-Control", "no-store, no-cache")
			err = parsedTemplate.Execute(w, p)
		}
	} else {
		http.NotFound(w, nil)
	}

}

func main() {
	fmt.Println("Vulnapp server listening : 1337")

	http.Handle("/assets/", http.StripPrefix("/assets/", http.FileServer(http.Dir("assets/"))))

	http.HandleFunc("/adminusers/", ShowAdminPageCache)
	err := http.ListenAndServe(":1337", nil)
	if err != nil {
		log.Fatal("ListenAndServe: ", err)
	}
}

```

## Example
Secure code example: The server is configured with strict cache controls and URL validation, preventing caching of dynamic or sensitive pages regardless of their URL pattern.


```go
package good

import (
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os/exec"
	"strings"
	"sync"
)

var sessionMap = make(map[string]string)

var (
	templateCache = make(map[string]*template.Template)
	mutex         = &sync.Mutex{}
)

type Lists struct {
	Uid       string
	UserName  string
	UserLists []string
	ReadFile  func(filename string) string
}

func parseTemplateFile(templateName string, tmplFile string) (*template.Template, error) {
	mutex.Lock()
	defer mutex.Unlock()

	// Check if the template is already cached
	if cachedTemplate, ok := templateCache[templateName]; ok {
		fmt.Println("cached")
		return cachedTemplate, nil
	}

	// Parse and store the template in the cache
	parsedTemplate, _ := template.ParseFiles(tmplFile)
	fmt.Println("not cached")

	templateCache[templateName] = parsedTemplate
	return parsedTemplate, nil
}

func ShowAdminPageCache(w http.ResponseWriter, r *http.Request) {

	if r.Method == "GET" {
		fmt.Println("cache called")
		sessionMap[r.RequestURI] = "admin"

		// Check if a session value exists
		if _, ok := sessionMap[r.RequestURI]; ok {
			cmd := "mysql -h mysql -u root -prootwolf -e 'select id,name,mail,age,created_at,updated_at from vulnapp.user where name not in (\"" + "admin" + "\");'"

			// mysql -h mysql -u root -prootwolf -e 'select id,name,mail,age,created_at,updated_at from vulnapp.user where name not in ("test");--';echo");'
			fmt.Println(cmd)

			res, err := exec.Command("sh", "-c", cmd).Output()
			if err != nil {
				fmt.Println("err : ", err)
			}

			splitedRes := strings.Split(string(res), "\n")

			p := Lists{Uid: "1", UserName: "admin", UserLists: splitedRes}

			parsedTemplate, _ := parseTemplateFile("page", "./views/admin/userlists.gtpl")
			w.Header().Set("Cache-Control", "no-store, no-cache")
			err = parsedTemplate.Execute(w, p)
		}
	} else {
		http.NotFound(w, nil)
	}

}

func main() {
	fmt.Println("Vulnapp server listening : 1337")

	http.Handle("/assets/", http.StripPrefix("/assets/", http.FileServer(http.Dir("assets/"))))

	http.HandleFunc("/adminusers", ShowAdminPageCache)
	err := http.ListenAndServe(":1337", nil)
	if err != nil {
		log.Fatal("ListenAndServe: ", err)
	}
}

```

## Example
Vulnerable code example: The server is configured with strict cache controls and URL validation, preventing caching of dynamic or sensitive pages regardless of their URL pattern.


```go
package fiber

import (
	"fmt"
	"log"

	"github.com/gofiber/fiber/v2"
)

func main() {
	app := fiber.New()
	log.Println("We are logging in Golang!")

	// GET /api/register
	app.Get("/api/*", func(c *fiber.Ctx) error {
		msg := fmt.Sprintf("✋")
		return c.SendString(msg) // => ✋ register
	})

	app.Post("/api/*", func(c *fiber.Ctx) error {
		msg := fmt.Sprintf("✋")
		return c.SendString(msg) // => ✋ register
	})

	// GET /flights/LAX-SFO
	app.Get("/flights/:from-:to", func(c *fiber.Ctx) error {
		msg := fmt.Sprintf("💸 From: %s, To: %s", c.Params("from"), c.Params("to"))
		return c.SendString(msg) // => 💸 From: LAX, To: SFO
	})

	// GET /dictionary.txt
	app.Get("/:file.:ext", func(c *fiber.Ctx) error {
		msg := fmt.Sprintf("📃 %s.%s", c.Params("file"), c.Params("ext"))
		return c.SendString(msg) // => 📃 dictionary.txt
	})

	log.Fatal(app.Listen(":3000"))
}

```

## Example
Vulnerable code example: The server is configured with strict cache controls and URL validation, preventing caching of dynamic or sensitive pages regardless of their URL pattern.


```go
package main

import (
	"net/http"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
)

func main() {
	r := chi.NewRouter()
	r.Use(middleware.Logger)
	r.Get("/*", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("welcome"))
	})
	http.ListenAndServe(":3000", r)
}

```

## Example
Vulnerable code example: The server is configured with strict cache controls and URL validation, preventing caching of dynamic or sensitive pages regardless of their URL pattern.


```go
package httprouter

import (
	"fmt"
	"log"
	"net/http"

	"github.com/julienschmidt/httprouter"
)

func Index(w http.ResponseWriter, r *http.Request, _ httprouter.Params) {
	fmt.Fprint(w, "Welcome!\n")
}

func Hello(w http.ResponseWriter, r *http.Request, ps httprouter.Params) {
	fmt.Fprintf(w, "hello, %s!\n", ps.ByName("name"))
}

func main() {
	router := httprouter.New()
	router.GET("/test/*test", Index)
	router.GET("/hello/:name", Hello)

	log.Fatal(http.ListenAndServe(":8082", router))
}

```

## References
* OWASP Web Cache Deception Attack: [Understanding Web Cache Deception Attacks](https://owasp.org/www-community/attacks/Web_Cache_Deception)
