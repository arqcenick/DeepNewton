using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SceneGenerator : MonoBehaviour {


    public GameObject ballPrefab;
    public GameObject obstaclePrefab;

    int numOfBalls;
    int numOfObstacles;

    List<BallScript> balls;

	// Use this for initialization
	void Start () {
        GenerateScene();


    }

    void GenerateScene()
    {
        numOfBalls = Mathf.CeilToInt(Random.Range(0f, 5f));
        numOfObstacles = Mathf.FloorToInt(Random.Range(0, 3f));

        for(int i = 0; i < numOfBalls; i++)
        {

            float xLoc = Random.Range(-4f, 4f);
            float zLoc = Random.Range(-4, 4f);
            Vector3 ballLoc = new Vector3(xLoc, 0, zLoc);

            var tempBall = Instantiate<GameObject>(ballPrefab, ballLoc, ballPrefab.transform.rotation);
            //balls.Add(tempBall.GetComponent<BallScript>());
        }

        for(int j = 0; j < numOfObstacles; j++)
        {

        }

    }
	
	// Update is called once per frame
	void Update () {


		
	}
}
