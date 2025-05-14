# Depending upon JCenter/Bintray as an artifact repository
[Bintray and JCenter are shutting down on February 1st, 2022](https://jfrog.com/blog/into-the-sunset-bintray-jcenter-gocenter-and-chartcenter/). Relying upon repositories that are deprecated or scheduled to be shutdown can have unintended consequences; for example, artifacts being resolved from a different artifact server or a total failure of the CI build.

When artifact repositories are left unmaintained for a long period of time, vulnerabilities may emerge. Theoretically, this could allow attackers to inject malicious code into the artifacts that you are resolving and infect build artifacts that are being produced. This can be used by attackers to perform a [supply chain attack](https://en.wikipedia.org/wiki/Supply_chain_attack) against your project's users.


## Recommendation
Always use the canonical repository for resolving your dependencies.


## Example
The following example shows locations in a Maven POM file where artifact repository upload/download is configured. The use of Bintray in any of these locations is not advised.


```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/maven-v4_0_0.xsd">

    <modelVersion>4.0.0</modelVersion>

    <groupId>com.semmle</groupId>
    <artifactId>parent</artifactId>
    <version>1.0</version>
    <packaging>pom</packaging>

    <name>Bintray Usage</name>
    <description>An example of using bintray to download and upload dependencies</description>

    <distributionManagement>
        <repository>
            <id>jcenter</id>
            <name>JCenter</name>
            <!-- BAD! Don't use JCenter -->
            <url>https://jcenter.bintray.com</url>
        </repository>
        <snapshotRepository>
            <id>jcenter-snapshots</id>
            <name>JCenter</name>
            <!-- BAD! Don't use JCenter -->
            <url>https://jcenter.bintray.com</url>
        </snapshotRepository>
    </distributionManagement>
    <repositories>
        <repository>
            <id>jcenter</id>
            <name>JCenter</name>
            <!-- BAD! Don't use JCenter -->
            <url>https://jcenter.bintray.com</url>
        </repository>
    </repositories>
    <repositories>
        <repository>
            <id>jcenter</id>
            <name>JCenter</name>
            <!-- BAD! Don't use Bintray -->
            <url>https://dl.bintray.com/groovy/maven</url>
        </repository>
    </repositories>
    <pluginRepositories>
        <pluginRepository>
            <id>jcenter-plugins</id>
            <name>JCenter</name>
            <!-- BAD! Don't use JCenter -->
            <url>https://jcenter.bintray.com</url>
        </pluginRepository>
    </pluginRepositories>
</project>

```

## References
* JFrog blog: [ Into the Sunset on May 1st: Bintray, JCenter, GoCenter, and ChartCenter ](https://jfrog.com/blog/into-the-sunset-bintray-jcenter-gocenter-and-chartcenter/)
* Common Weakness Enumeration: [CWE-1104](https://cwe.mitre.org/data/definitions/1104.html).
