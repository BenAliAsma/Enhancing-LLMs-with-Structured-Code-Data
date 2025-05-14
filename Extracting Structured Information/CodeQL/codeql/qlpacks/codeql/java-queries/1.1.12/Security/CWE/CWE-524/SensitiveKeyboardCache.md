# Android sensitive keyboard cache
When a user enters information in a text input field on an Android application, their input is saved to a keyboard cache which provides autocomplete suggestions and predictions. There is a risk that sensitive user data, such as passwords or banking information, may be leaked to other applications via the keyboard cache.


## Recommendation
For input fields expected to accept sensitive information, use input types such as `"textNoSuggestions"` (or `"textPassword"` for a password) to ensure the input does not get stored in the keyboard cache.

Optionally, instead of declaring an input type through XML, you can set the input type in your code using `TextView.setInputType()`.


## Example
In the following example, the field labeled BAD allows the password to be saved to the keyboard cache, whereas the field labeled GOOD uses the `"textPassword"` input type to ensure the password is not cached.


```xml
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto">

    <!-- BAD: This password field uses the `text` input type, which allows the input to be saved to the keyboard cache. -->
    <EditText
        android:id="@+id/password_bad"
        android:inputType="text"/> 

    <!-- GOOD: This password field uses the `textPassword` input type, which ensures that the input is not saved to the keyboard cache. -->
    <EditText
        android:id="@+id/password_good"
        android:inputType="textPassword"/>  
</LinearLayout>
```

## References
* OWASP Mobile Application Security Testing Guide: [Determining Whether the Keyboard Cache Is Disabled for Text Input Fields](https://github.com/OWASP/owasp-mastg/blob/b7a93a2e5e0557cc9a12e55fc3f6675f6986bb86/Document/0x05d-Testing-Data-Storage.md#determining-whether-the-keyboard-cache-is-disabled-for-text-input-fields-mstg-storage-5).
* Android Developers: [android:inputType attribute documentation.](https://developer.android.com/reference/android/widget/TextView#attr_android:inputType)
* Common Weakness Enumeration: [CWE-524](https://cwe.mitre.org/data/definitions/524.html).
