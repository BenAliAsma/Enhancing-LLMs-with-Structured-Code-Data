# Access Java object methods through JavaScript exposure
Calling the `addJavascriptInterface` method of the `android.webkit.WebView` class allows the web pages of a WebView to access a Java object's methods via JavaScript.

Objects exposed to JavaScript are available in all frames of the WebView.


## Recommendation
If you need to expose Java objects to JavaScript, guarantee that no untrusted third-party content is loaded into the WebView.


## Example
In the following (bad) example, a Java object is exposed to JavaScript.


```java
import android.webkit.JavascriptInterface;
import android.database.sqlite.SQLiteOpenHelper;

class ExposedObject extends SQLiteOpenHelper {
    @JavascriptInterface
    public String studentEmail(String studentName) {
        // SQL injection
        String query = "SELECT email FROM students WHERE studentname = '" + studentName + "'";

        Cursor cursor = db.rawQuery(query, null);
        cursor.moveToFirst();
        String email = cursor.getString(0);

        return email;
    }
}

webview.getSettings().setJavaScriptEnabled(true);
webview.addJavaScriptInterface(new ExposedObject(), "exposedObject");
webview.loadData("", "text/html", null);

String name = "Robert'; DROP TABLE students; --";
webview.loadUrl("javascript:alert(exposedObject.studentEmail(\""+ name +"\"))");

```

## References
* Android Documentation: [addJavascriptInterface](https://developer.android.com/reference/android/webkit/WebView#addJavascriptInterface(java.lang.Object,%20java.lang.String))
* Common Weakness Enumeration: [CWE-79](https://cwe.mitre.org/data/definitions/79.html).
