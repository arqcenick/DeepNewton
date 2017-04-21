using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Capture : MonoBehaviour {
    public string folder = "ScreenshotFolder2";
    public int frameRate = 25;


    // Use this for initialization
    void Start () {
        Time.captureFramerate = 8;

        // Create the folder
        System.IO.Directory.CreateDirectory(folder);
    }
	
	// Update is called once per frame
	void Update () {
        string name = string.Format("{0}/{1:D04}_shot.png", folder, Time.frameCount);
        Application.CaptureScreenshot(name);
    }
}
