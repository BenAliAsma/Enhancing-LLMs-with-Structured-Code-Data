# Insecure Spring Boot Actuator Configuration
Spring Boot is a popular framework that facilitates the development of stand-alone applications and micro services. Spring Boot Actuator helps to expose production-ready support features against Spring Boot applications.

Endpoints of Spring Boot Actuator allow to monitor and interact with a Spring Boot application. Exposing unprotected actuator endpoints through configuration files can lead to information disclosure or even remote code execution vulnerability.

Rather than programmatically permitting endpoint requests or enforcing access control, frequently developers simply leave management endpoints publicly accessible in the application configuration file `application.properties` without enforcing access control through Spring Security.


## Recommendation
Declare the Spring Boot Starter Security module in XML configuration or programmatically enforce security checks on management endpoints using Spring Security. Otherwise accessing management endpoints on a different HTTP port other than the port that the web application is listening on also helps to improve the security.


## Example
The following examples show both 'BAD' and 'GOOD' configurations. In the 'BAD' configuration, no security module is declared and sensitive management endpoints are exposed. In the 'GOOD' configuration, security is enforced and only endpoints requiring exposure are exposed.


```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>spring-boot-actuator-app</groupId>
    <artifactId>spring-boot-actuator-app</artifactId>
    <version>1.0-SNAPSHOT</version>

    <properties>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
        <maven.compiler.source>1.8</maven.compiler.source>
        <maven.compiler.target>1.8</maven.compiler.target>
    </properties>

    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>2.3.8.RELEASE</version>
        <relativePath/>
    </parent>

    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-actuator</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-devtools</artifactId>
        </dependency>

        <!-- GOOD: Enable Spring Security -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-security</artifactId>
        </dependency>

        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-test</artifactId>
        </dependency>
    </dependencies>

</project>
```

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>spring-boot-actuator-app</groupId>
    <artifactId>spring-boot-actuator-app</artifactId>
    <version>1.0-SNAPSHOT</version>

    <properties>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
        <maven.compiler.source>1.8</maven.compiler.source>
        <maven.compiler.target>1.8</maven.compiler.target>
    </properties>

    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>2.3.8.RELEASE</version>
        <relativePath/>
    </parent>

    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-actuator</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-devtools</artifactId>
        </dependency>

        <!-- BAD: No Spring Security enabled -->
        <!-- dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-security</artifactId>
        </dependency -->

        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-test</artifactId>
        </dependency>
    </dependencies>

</project>
```

```none
#management.endpoints.web.base-path=/admin


#### BAD: All management endpoints are accessible #### 
# vulnerable configuration (spring boot 1.0 - 1.4): exposes actuators by default

# vulnerable configuration (spring boot 1.5+): requires value false to expose sensitive actuators
management.security.enabled=false

# vulnerable configuration (spring boot 2+): exposes health and info only by default, here overridden to expose everything
management.endpoints.web.exposure.include=*


#### GOOD: All management endpoints have access control #### 
# safe configuration (spring boot 1.0 - 1.4): exposes actuators by default
management.security.enabled=true

# safe configuration (spring boot 1.5+): requires value false to expose sensitive actuators
management.security.enabled=true

# safe configuration (spring boot 2+): exposes health and info only by default, here overridden to expose one additional endpoint which we assume is intentional and safe.
management.endpoints.web.exposure.include=beans,info,health

```

## References
* Spring Boot documentation: [Spring Boot Actuator: Production-ready Features](https://docs.spring.io/spring-boot/docs/current/reference/html/production-ready-features.html)
* VERACODE Blog: [Exploiting Spring Boot Actuators](https://www.veracode.com/blog/research/exploiting-spring-boot-actuators)
* HackerOne Report: [Spring Actuator endpoints publicly available, leading to account takeover](https://hackerone.com/reports/862589)
* Common Weakness Enumeration: [CWE-16](https://cwe.mitre.org/data/definitions/16.html).
