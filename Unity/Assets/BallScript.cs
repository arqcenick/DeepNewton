﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BallScript : MonoBehaviour {


    Rigidbody rigid;
    float time = 10f;
    Vector2 vec;



    float power;
    private void Awake()
    {
         
    }

    // Use this for initialization
    void Start () {

        vec = Random.insideUnitCircle;
        power = 20000f + Random.value * 20000f;
        rigid = gameObject.GetComponent<Rigidbody>();
        rigid.AddForce(new Vector3(vec.x * power, 0f, vec.y * power));
        
        
	}
	
	// Update is called once per frame
	void Update () {
        time -= Time.deltaTime;
        if (time < 0)
        {
            UnityEditor.EditorApplication.isPlaying = false;

        }
	}
}
