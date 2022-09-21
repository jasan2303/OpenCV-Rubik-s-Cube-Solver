/// main.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include "rubiks.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <syslog.h>

#define ESCAPE_KEY (27)

using namespace cv;
using namespace std;

struct timespec start_time, cur_time, delta_time;

vector<Point> centers;

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle(Point pt1, Point pt2, Point pt0) {

    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);

}


// displays the identified squares onto the screen
static void drawSquares(Mat &image, const vector<vector<Point>> &squares)
{

   for (size_t i = 0; i < squares.size(); i++) {
        const Point *p1 = &squares[i][0];
        const Point *p2 = &squares[i][2];

        Rect r = boundingRect(Mat(squares[i]));
        Mat roi = image(r);
        Scalar color = mean(roi); 
       
        rectangle( image, *p1, *p2, color, 5, LINE_AA);
   }
 
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
static void findSquares(const Mat &image, vector<vector<Point>> &squares, bool inv = false) {

    squares.clear();
    Mat gray,gray0,hsv;
    vector<vector<Point>> contours;

    //cvtColor(image,hsv,COLOR_BGR2HSV);
    //Mat hsv_channels[3];
    //cv::split(hsv, hsv_channels); // splits the hsv values into it's own array
    //gray0 = hsv_channels[2]; // the 3rd channel is a grayscale value
       cvtColor(image, gray0, COLOR_BGR2GRAY);

 // GaussianBlur(gray0, gray0, Size(7,7), 1.5, 1.5);
   // Canny(gray0, gray, 0, 30, 3);
    Canny(gray0, gray, 150, 250, 5);
    // find contours and store them all as a list
    findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    vector<Point> approx;

    // test each contour
    for( size_t i = 0; i < contours.size(); i++) {
        // approximate contour with accuracy proportional
        // to the contour perimeter
        approxPolyDP(Mat(contours[i]), approx, 5, true);

        // square contours should have 4 vertices after approximation
        // relatively large area (to filter out noisy contours)
        // and be convex.
        // Note: absolute value of an area is used because
        // area may be positive or negative - in accordance with the
        // contour orientation
        if( approx.size() == 4 &&
               // fabs(contourArea(Mat(approx))) > 10 &&
                fabs(contourArea(approx)) > 500 && fabs(contourArea(approx)) < 5000 &&
                isContourConvex(Mat(approx))) {
            double maxCosine = 0;

            for( int j = 2; j < 5; j++ ) {

                // find the maximum cosine of the angle between joint edges
                double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
                maxCosine = MAX(maxCosine, cosine);

            }

            // if cosines of all angles are small
            // (all angles are ~90 degree) then write quandrange
            // vertices to resultant sequence
            if( maxCosine < 0.1 && squares.size() <= 18) squares.push_back(approx);
        }

    }

}

//function to select 9 blocks of a face
int blocks_filter(vector<vector<Point> >& list)
{

    int N = list.size();
    vector<int> concern_sqrs(N, 1);
    
    


    for( size_t i = 0; i < N ; i++ )
    { 

        Point* p1 = &list[i][0];
        Point* p2 = &list[i][2];
        
        Point centroid;
        centroid.x = (p1->x + p2->x)/2;
        centroid.y = (p1->y + p2->y)/2;        
        centers.push_back(centroid);
        
    } 
    
    //cout<<" No of senters "<<centers.size()<<endl;
 /*  cout<<"centers before filter "<<endl;
   for(size_t i=0; i< centers.size();i++)
   { 
      cout<<"center  X ="<<centers[i].x<<" Y = "<<centers[i].y<<endl;
   }
   */ 
    for( size_t i = 0; i < N ; i++ )
    { 
        if(concern_sqrs[i])
        {
          for(size_t j=i+1; j < N ; j++ )
          {
            
            

              Point diff_point;
              diff_point.x = abs(centers[i].x - centers[j].x);  diff_point.y = abs(centers[i].y - centers[j].y);
              if( ( diff_point.x < 3) &&  ( diff_point.y < 3)  )
              {
                 //remove this square from the   
                 concern_sqrs[j] = 0;
              }
             
          }
        }
    }

   /*
    for( size_t i = 0; i < N ; i++ )
    { 
      //  if(concern_sqrs[i])
       // {
          for(size_t j=0; j < list.size() ; j++ )
          {
            if( i != j){

            
              Point* p1 = &list[i][1];
              Point* p2 = &list[j][1] ;
              Point diff_point;
              diff_point.x = p1->x - p2->x;  diff_point.y = p1->y - p2->y;
              if( ( abs(diff_point.x) < 11) &&  ( abs(diff_point.y) < 11)  )
              {
                 //remove this square from the   
                 concern_sqrs[j] = 0;
              }
           }
         }
        //}
    }
    */
 /*  for( size_t i=0; i< N; i++)
   {
      cout<<" "<<concern_sqrs[i];
   }
   cout<<endl; */
   int erased=0;
   for( size_t i = 0; i < N ; i++ )
   {
      if( !concern_sqrs[i] )
      {
         Point* p1 = &list[i][0];
         //cout<<" X= "<<p1->x<<" Y= "<<p1->y<<endl;

         list.erase(list.begin() + (i - erased));
         centers.erase(centers.begin() + (i - erased));
         erased++;
      }

   }
  /* cout<<"filtered centers"<<endl;
    for(size_t j=0; j < centers.size() ; j++ )
    {
          
       cout<<"center  X= "<<centers[j].x<<" Y= "<<centers[j].y<<endl;

     } */
   
   N = centers.size();
   if( N == 9)
    return 1;
   else
    return 0;
}

//globalizing

Scalar colors[6] = {
		{70, 25, 130, 0}, // red   {70, 25, 130, 0}, {25, 25, 205, 0}
		{90, 90, 180, 0}, // orange  {90, 90, 180, 0} {10, 110, 255, 0}
		{100, 185, 185, 0}, // yellow
		{65, 95, 35, 0}, // green
		{140, 25, 0, 0}, // blue
		{200, 175, 160, 0} // white

	};



static char detectBlockColor(Scalar color) {
  vector<Rect> rects;
      /*    colors[0] = 		{70, 25, 130, 0};	 
  
		//{70, 25, 130, 0}, // red   {70, 25, 130, 0}, {25, 25, 205, 0}
	colors[1] =	{90, 90, 180, 0}; // orange  {90, 90, 180, 0} {10, 110, 255, 0}
	colors[2] =	{100, 185, 185, 0};// yellow
	colors[3] =	{65, 95, 35, 0}; // green
	colors[4] =	{140, 25, 0, 0}; // blue
	colors[5] =	{200, 175, 160, 0}; // white
     */

  char colorchars[6] = {
    'R', 'O', 'Y',
    'G', 'B', 'W'

  };


    Scalar tempcolor = {0, 0, 0, 0};
    char c = ' ';
    int min = 1000;
    for (size_t r = 0; r < 6; r++) { // 6 colors to check
      tempcolor = colors[r]; // gets the current colors
      int delta = fabs(tempcolor[0] - color[0])
        + fabs(tempcolor[1] - color[1])
        + fabs(tempcolor[2] - color[2]);

      // cout << delta << endl;
      if (delta < min) {
        min = delta;
        c = colorchars[r];

      }

    }

    
  //cout<<"Face Color "<<c<<endl;
   return c;
}


static char FaceData[6][9];

int FaceRecord(Mat &image, vector<vector<Point>> &squares)
{
  static char Faces[6] = { 0,0,0,0,0,0 };   // { R, O, Y, G, B, W} 
  static int x=0;
char winInput;
 // char rubik[] = { 'R', 'O', 'Y', 'G', 'B', 'W'};
//char rubik[] = { 'R', 'O', 'W', 'Y', 'B', 'G'};
  char rubik[] = { 'W', 'Y', 'G', 'R', 'O', 'B'};
   char face_c, c;
    Rect r = boundingRect(Mat(squares[4]));   //probably the center block is 4
    Mat roi = image(r);
    Scalar color = mean(roi);
    face_c = detectBlockColor( color);
  //cout<<"Face="<<FaceColor(x)<<endl;
  if( face_c == rubik[x] ) 
  { 
    cout<<"Face color "<<face_c<<" is being recorded\n";
    
   

   for( size_t i=0; i< 9; i++)
   {
     Rect r = boundingRect(Mat(squares[i]));   //probably the center block is 4
     Mat roi = image(r);
     Scalar color = mean(roi);
     FaceData[x][i] = detectBlockColor( color);
   }
   cout<<"Face color "<<face_c<<" is recorded, change the face\n";
        /* while(1)
         {  winInput = (char)waitKey(1);
          if(  winInput  == (char)ESCAPE_KEY)
           { break; }
         }*/
   x++; //moving to next color
  }
  return x;
}




/// <summary>
/// Takes input of face color and returns enum value of that face
/// </summary >
/// <param name="c">Input character of face color</param>
/// <returns>enum eColor value of that face</returns>
eColor getColor(char c)
{
    switch (c)
    {
    case 'W': return WHITE;
    case 'Y': return YELLOW;
    case 'B': return BLUE;
    case 'G': return GREEN;
    case 'R': return RED;
    case 'O': return ORANGE;
    }
}

/// <summary>
/// This function converts enum eColor to string and returns to printCube function
/// </summary>
/// <param name="c">: enum eColor type input of face colors</param>
/// <returns>Character of color</returns>
char getColorCharacter(int c)
{
    switch (c)
    {
    case WHITE: return 'W';
    case YELLOW: return 'Y';
    case BLUE: return 'B';
    case GREEN: return 'G';
    case RED: return 'R';
    case ORANGE: return 'O';
    }
    exit(0);
}

/// <summary>
/// Inputs a vector element of moveList given by getMoveListString function and converts it to string
/// </summary>
/// <param name="m">: An element of vector moveList</param>
/// <returns>String of face moves</returns>
string getMoveString(const eMove& m)
{
    switch (m)
    {
    case U: return "U";
    case D: return "D";
    case F: return "F";
    case B: return "B";
    case L: return "L";
    case R: return "R";
    case U2: return "U2";
    case D2: return "D2";
    case F2: return "F2";
    case B2: return "B2";
    case L2: return "L2";
    case R2: return "R2";
    }
    return "";
}



/// <summary>
/// To optimise the solution of solving a Rubik's cube by removing the unnecessary/extra moves
/// </summary>
/// <param moveString">: A string consisting of all the stages to solve a cube</param>
/// <returns>String of optimised moves</returns>

std::string optimiseMoves(std::string& moveString) {
    int start = 0;
    std::string prevMove, optimizedString;

    for (int i = 0; i < moveString.size(); i++) {
        if (moveString[i] == ' ') {
            auto Currentmove = moveString.substr(start, i - start); //creating substrings of all the moves

            if (prevMove[0] == Currentmove[0]) { //Checking for the 0th index of the 2 moves for the face (L, R, F, B, U or D)
                prevMove.clear();

                if (Currentmove[1] == '2' || prevMove[1] == '2') { //For cases like L L2 (L L2 = L')
                    optimizedString.push_back(Currentmove[0]);
                    optimizedString = optimizedString + "\' ";
                }
                else {
                    prevMove.push_back(Currentmove[0]); //For cases like L L (L L = L2)
                    prevMove = prevMove + "2 ";
                }
            }
            else {
                optimizedString = optimizedString + prevMove;
                prevMove = Currentmove + ' ';
            }
            start = i + 1;
        }
    }
    return optimizedString + prevMove;
}


/// <summary>
/// Converts vector list of moves into string
/// </summary>
/// <param name="moveList">: List of moves generated by different stages</param>
/// <returns>String sequence of moves</returns>
string getMoveListString(const vector<eMove>& moveList)
{
    string moveSequenceString;
    for (auto& m : moveList)                               //Traversing all the elements of vector moveList
    {
        moveSequenceString += getMoveString(m) + ' ';
    }
    return moveSequenceString;
}

/// <summary>
/// This function prints the cube
/// </summary>
/// <param name="faces">: FaceArray type input of faces</param>
/// <param name="centers">: Array input of centers</param>
void printCube(const FaceArray& faces, eColor centres[]) {
    std::cout << std::endl;
    std::cout << "-----Printing Cube-----" << std::endl;
    
    for (int i = 0; i < 6; ++i) {                          //Iterating loop for 6 faces
        char c[9];
        uint_fast32_t face = faces[i];                     //Storing a face of cube in variable face

        c[3] = getColorCharacter(face & 0xF);              //Extracting colors from face variable       
        face = face >> 4;
        c[6] = getColorCharacter(face & 0xF);
        face = face >> 4;
        c[7] = getColorCharacter(face & 0xF);
        face = face >> 4;
        c[8] = getColorCharacter(face & 0xF);
        face = face >> 4;
        c[5] = getColorCharacter(face & 0xF);
        face = face >> 4;
        c[2] = getColorCharacter(face & 0xF);
        face = face >> 4;
        c[1] = getColorCharacter(face & 0xF);
        face = face >> 4;
        c[0] = getColorCharacter(face & 0xF);
        c[4] = getColorCharacter(centres[i]);

        for (int j = 0; j < 9; ++j)                        //Printing the Extracted colors
            std::cout << c[j];

        std::cout << std::endl;
    }
    std::cout << "-----------------------" << std::endl;
}

/// <summary>
/// This Function will take input of faces from the user
/// </summary>
/// <param name="faces">Array of type FaceArray</param>
/// <param name="centers">enum eColor for storing center faces</param>
/*
void readData(FaceArray& faces, eColor centers[])
{
    int i, j;
    cout << "Type Faces in the order UDFBLR:" << endl;
    for (int i = 0;i < 6;i++)                   //loop for all 6 faces of cube
    {
        char c[9];
        //Taking input of 9 face colors in the form W,Y,G,R,O,B where each represents White, Yellow, Green, Red, Orange, Blue color
        cin >> c[0] >> c[1] >> c[2] >> c[3] >> c[4] >> c[5] >> c[6] >> c[7] >> c[8];
        faces[i] <<= 4;                      //Left shifting by 4 bytes to accomodate all faces in 32 bytes.
        faces[i] |= getColor(c[0]);         //Applying Bitwise OR Operation to store color enum value in faces
        faces[i] <<= 4;
        faces[i] |= getColor(c[1]);
        faces[i] <<= 4;
        faces[i] |= getColor(c[2]);
        faces[i] <<= 4;
        faces[i] |= getColor(c[5]);
        faces[i] <<= 4;
        faces[i] |= getColor(c[8]);
        faces[i] <<= 4;
        faces[i] |= getColor(c[7]);
        faces[i] <<= 4;
        faces[i] |= getColor(c[6]);
        faces[i] <<= 4;
        faces[i] |= getColor(c[3]);

        centers[i] = getColor(c[4]);        //Storing enum value of center color in centers array
    }
}*/
/*
void readData(FaceArray& faces, eColor centers, char FaceData[][])
{
   int i,j;

   cout<<" Collected face data is being feed to the algorithm\n";
   j=0;
   for(int i=0; i<6;i++)
   {
      faces[i] <<= 4;                      //Left shifting by 4 bytes to accomodate all faces in 32 bytes.
        faces[i] |= getColor(FaceData[j][0]);         //Applying Bitwise OR Operation to store color enum value in faces
        faces[i] <<= 4;
        faces[i] |= getColor(FaceData[j][1]);
        faces[i] <<= 4;
        faces[i] |= getColor(FaceData[j][2]);
        faces[i] <<= 4;
        faces[i] |= getColor(FaceData[j][5]);
        faces[i] <<= 4;
        faces[i] |= getColor(FaceData[j][8]);
        faces[i] <<= 4;
        faces[i] |= getColor(FaceData[j][7]);
        faces[i] <<= 4;
        faces[i] |= getColor(FaceData[j][6]);
        faces[i] <<= 4;
        faces[i] |= getColor(FaceData[j][3]);

        centers[i] = getColor(FaceData[j][4]);  
        j++;
   }

}
*/


int colorCalibration(Mat &image, vector<vector<Point>> squares)
{
   static int flag[] = { 0,0,0,0,0,0};
   static int print_flag[] = { 0,0,0,0,0,0};
   char winInput;
   static int callibrated=0;

   if(flag[0])
   { 
     if(!print_flag[0])
     {    cout<<"White color is calibrated change the face to Yellow\n";  print_flag[0] = 1;  
         while(1)
         {  winInput = (char)waitKey(1);
          if(  winInput  == (char)ESCAPE_KEY)
           { break; }
         }
           goto down;
     }
   }
   else
   {     
     Rect r = boundingRect(Mat(squares[4]));   //probably the center block is 4
     Mat roi = image(r);
      colors[5] = mean(roi);
     flag[0] =1; 
     goto down;
   }
   
   if(flag[1])
   { 
      if(!print_flag[1])
     {  cout<<"Yellow is calibrated change the face to Green \n"; print_flag[1] = 1;  
          while(1)
         {  winInput = (char)waitKey(1);
          if(  winInput  == (char)ESCAPE_KEY)
           { break; }
         }
           goto down;
     }
   }
   else if( flag[0] )
   {
      Rect r = boundingRect(Mat(squares[4]));   //probably the center block is 4
     Mat roi = image(r);
      colors[2] = mean(roi);
     flag[1] =1; 
     goto down;
   }

   if(flag[2])
   { 
      if(!print_flag[2])
     {  
        cout<<"green is calibrated change the face to red \n"; print_flag[2] = 1;  
          while(1)
         {  winInput = (char)waitKey(1);
          if(  winInput  == (char)ESCAPE_KEY)
           { break; }
         }
           goto down;
     }
   }
   else if( flag[0] && flag[1])
   {
      Rect r = boundingRect(Mat(squares[4]));   //probably the center block is 4
     Mat roi = image(r);
      colors[3] = mean(roi);
     flag[2] =1; 
     goto down;
   }
   
   if(flag[3])
   {
     if(!print_flag[3])
     {   cout<<"red is calibrated change the face to orange\n"; print_flag[3] = 1;  
          while(1)
         {  winInput = (char)waitKey(1);
          if(  winInput  == (char)ESCAPE_KEY)
           { break; }
         }
           goto down;
     }
   }
   else if( flag[0] && flag[1] && flag[2] )
   {
      Rect r = boundingRect(Mat(squares[4]));   //probably the center block is 4
     Mat roi = image(r);
      colors[0] = mean(roi);
     flag[3] =1; 
     goto down;
   }
     
   if(flag[4])
   { 
      if(!print_flag[4])
     {  
        cout<<"orange is calibrated change the face anfd prees esc\n"; print_flag[4] = 1;  
          while(1)
         {  winInput = (char)waitKey(1);
          if(  winInput  == (char)ESCAPE_KEY)
           { break; }
         }
           goto down;
     }
   }
   else if( flag[0] && flag[1] && flag[2] && flag[3] )
   {
      Rect r = boundingRect(Mat(squares[4]));   //probably the center block is 4
     Mat roi = image(r);
      colors[1] = mean(roi);
     flag[4] =1; 
     goto down;
   }

   if(flag[5])
   { 
     cout<<"Blue is calibrated  \n CALLIBRATION COMPLETE \n";
     callibrated=1; 
     /*Scalar  AVGcolors[6] = {
		{70, 25, 130, 0}, // red   {70, 25, 130, 0}, {25, 25, 205, 0}
		{90, 90, 180, 0}, // orange  {90, 90, 180, 0} {10, 110, 255, 0}
		{100, 185, 185, 0}, // yellow
		{65, 95, 35, 0}, // green
		{140, 25, 0, 0}, // blue
		{200, 175, 160, 0} // white

	}; 

   colors[0] = (50*colors[0] + 50*AVGcolors[0])/100;
   colors[1] = (50*colors[1] + 50 *AVGcolors[1])/100;
   colors[2] = (50*colors[2] + 50*AVGcolors[2])/100;
   colors[3] = (50*colors[3] + 50*AVGcolors[3])/100;
   colors[4] = (50*colors[4] + 50*AVGcolors[4])/100;
   colors[0] = (50*colors[5] + 50*AVGcolors[5])/100;
   */}
   else if( flag[0] && flag[1] && flag[2] && flag[3] && flag[4] )
   {
      Rect r = boundingRect(Mat(squares[4]));   //probably the center block is 4
     Mat roi = image(r);
      colors[4] = mean(roi);
     flag[5] =1; 
     goto down;
   }
   
  down:
   if(callibrated)
    return 1;
   else 
    return 0;
}

int main()
{
      static int frame_count=0;
      static int callib_done =0;
Mat frame;
    vector<vector<Point>> squares;
    VideoCapture cap(0);
    static int face_record_count=0; 
    char c;
    //noting down time before frame capture starts
    clock_gettime(CLOCK_MONOTONIC, &start_time); //to run the capture for 60sec
    start_time.tv_sec += 60;
     int i=0; char buffer[25];
    cout<<" \n\n OPENCV RUBIK'S SOLVER \n\n";
    cout<<" Show the faces of the cube inte order: \n";
    cout<<"   WHITE \n   YELLOW \n   GREEN \n   RED \n   ORANGE \n   BLUE \n";
    cap.set(CAP_PROP_FRAME_WIDTH, 320);
   cap.set(CAP_PROP_FRAME_HEIGHT, 240);
    while (1)
    {
      
       clock_gettime(CLOCK_MONOTONIC, &cur_time);
       if( cur_time.tv_sec <= start_time.tv_sec)
          frame_count++;

      centers.clear();
      cap >> frame;
      if (frame.empty()) return -1;

      findSquares(frame, squares);
      // output_squares(squares);
      int N = squares.size();
     // cout<<"N = "<<N<<endl;
      if(blocks_filter(squares)) 
      {
            if(!callib_done)
            {
              callib_done = colorCalibration(frame, squares);
            }
            if(callib_done)
            {
               face_record_count = FaceRecord( frame, squares);
               drawSquares(frame, squares);
               if(face_record_count == 6)
               {  goto Faces_collected; }
            }
      }
       imshow("Rubic Detection Demo", frame);
       drawSquares(frame, squares);

       sprintf(buffer,"frame%04d.png",++i);
     //  imwrite(buffer, frame);
      //imshow("Rubic all squares", frame);
     // imwrite("test.png", frame);
      waitKey(1);
       
    }

   
    Faces_collected: cout<<"Frame rate = "<<frame_count/60<<"\n";
    destroyWindow("Rubic Detection Demo");
   //  destroyWindow("Rubic all squares");
    cout<<"Data collected:"<<endl;

    static char FaceDataCrc[6][9];
    int z=0;
    for( size_t i=0;i<6; i++)
    {
        FaceDataCrc[i][0] = FaceData[i][8];
        FaceDataCrc[i][1] = FaceData[i][7];
        FaceDataCrc[i][2] = FaceData[i][6];
        FaceDataCrc[i][3] = FaceData[i][5];
        FaceDataCrc[i][4] = FaceData[i][4];
        FaceDataCrc[i][5] = FaceData[i][3];
        FaceDataCrc[i][6] = FaceData[i][2];
        FaceDataCrc[i][7] = FaceData[i][1];
        FaceDataCrc[i][8] = FaceData[i][0];
    }
    for(size_t i=0; i<6; i++)
    {
      for(size_t j=0; j<9; j++)
      {  cout<<" "<<FaceDataCrc[i][j]<<" "; }  // cout<<" "<<FaceData[i][8 - j]<<" "; 
      cout<<endl;
    }
    FaceArray faces = { 0,0,0,0,0,0 };          //Initializing array for storing scrambled faces
    eColor centers[6];                          //Initializing array for storing Colours at middle cubie of the face
    cout<<" Collected face data is being feed to the algorithm\n";
  int  j=0;
   for(int i=0; i<6;i++)
   {
      /*

      faces[i] <<= 4;                      //Left shifting by 4 bytes to accomodate all faces in 32 bytes.
        faces[i] |= getColor(FaceData[j][8]);         //Applying Bitwise OR Operation to store color enum value in faces
        faces[i] <<= 4;
        faces[i] |= getColor(FaceData[j][7]);
        faces[i] <<= 4;
        faces[i] |= getColor(FaceData[j][6]);
        faces[i] <<= 4;
        faces[i] |= getColor(FaceData[j][5]);
        faces[i] <<= 4;
        faces[i] |= getColor(FaceData[j][4]);
        faces[i] <<= 4;
        faces[i] |= getColor(FaceData[j][3]);
        faces[i] <<= 4;
        faces[i] |= getColor(FaceData[j][2]);
        faces[i] <<= 4;
        faces[i] |= getColor(FaceData[j][1]);

        centers[i] = getColor(FaceData[j][0]);  */

faces[i] <<= 4;                      //Left shifting by 4 bytes to accomodate all faces in 32 bytes.
        faces[i] |= getColor(FaceDataCrc[j][0]);         //Applying Bitwise OR Operation to store color enum value in faces
        faces[i] <<= 4;
        faces[i] |= getColor(FaceDataCrc[j][1]);
        faces[i] <<= 4;
        faces[i] |= getColor(FaceDataCrc[j][2]);
        faces[i] <<= 4;
        faces[i] |= getColor(FaceDataCrc[j][3]);
        faces[i] <<= 4;
        faces[i] |= getColor(FaceDataCrc[j][5]);
        faces[i] <<= 4;
        faces[i] |= getColor(FaceDataCrc[j][6]);
        faces[i] <<= 4;
        faces[i] |= getColor(FaceDataCrc[j][7]);
        faces[i] <<= 4;
        faces[i] |= getColor(FaceDataCrc[j][8]);

        centers[i] = getColor(FaceDataCrc[j][4]);  
        
   }
    //readData(faces, centers, FaceData);                   //Function that will take input of faces from the user

    std::cout << "Initialising..";
    initialiseSolver(centers);

    printCube(faces, centers);

    string moveString;

    auto moves = getStage1Moves(faces, centers);   //Applying IDDFS Algorithm on stage 1 moves
    auto stageString = getMoveListString(moves);   //Converting the moves returned by IDDFS to string 
    cout << "Stage 1 Moves: " << stageString;      //Printing the moves
    doMoveList(faces, moves);                      //Applying the moves on rubik's cube
    moveString += stageString;

    printCube(faces, centers);                      //Printing the cube

    moves = getStage2Moves(faces, centers);         //Applying IDDFS Algorithm on stage 2 moves
    stageString = getMoveListString(moves);         //Converting the moves returned by IDDFS to string 
    cout << "Stage 2 Moves: " << stageString;       //Printing the moves
    doMoveList(faces, moves);                       //Applying the moves on rubik's cube
    moveString += stageString;

    printCube(faces, centers);                       //Printing the cube


    moves = getStage3Moves(faces, centers);         //Applying IDDFS Algorithm on stage 3 moves
    stageString = getMoveListString(moves);         //Converting the moves returned by IDDFS to string 
    cout << "Stage 3 Moves: " << stageString;       //Printing the moves
    doMoveList(faces, moves);                       //Applying the moves on rubik's cube
    moveString += stageString;

    std::cout << "\n\nSolution: " << moveString << std::endl;
    moveString = optimiseMoves(moveString);
    std::cout << "Optimised: " << moveString << std::endl;

    int moveCount = 0;
    for (auto& c : moveString) {
        if (c == ' ')
            ++moveCount;
    }

    std::cout << "Moves Needed: " << moveCount << std::endl << std::endl;

    return 0;

}
