#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;



vector<Point> centers;


// rectangle comparator
static bool compareRects(Rect, Rect);

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
static void drawSquares(Mat &image, const vector<vector<Point>> &squares) {

    /*for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        int shift = 1;
        Rect r=boundingRect( Mat(squares[i]));
        r.x = r.x + r.width / 4;
        r.y = r.y + r.height / 4;
        r.width = r.width / 2;
        r.height = r.height / 2;
        Mat roi = image(r);
        Scalar color = mean(roi);
        polylines(image, &p, &n, 1, true, color, 2, LINE_AA, shift);
        Point center( r.x + r.width/2, r.y + r.height/2 );
        ellipse( image, center, Size( r.width/2, r.height/2), 0, 0, 360, color, 2, LINE_AA );
    }*/
   // cout<<"NO OF SQ="<<squares.size()<<endl;
  /*  for (size_t i = 0; i < squares.size(); i++) {

        const Point *p = &squares[i][0];
        int n = (int) squares[i].size();
        int shift = 1;
        Rect r = boundingRect(Mat(squares[i]));

      /*  if (r.width > 30 && r.height > 30) {
          r.x = r.x + r.width / 4;
          r.y = r.y + r.height / 4;
          r.width = r.width / 2;
          r.height = r.height / 2;
      
          Mat roi = image(r);
          Scalar color = mean(roi);
          polylines(image, &p, &n, 1, true, color, 2, LINE_AA, shift);

        }

      }  */

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

static void detectColors(const Mat &image, const vector<vector<Point>> &squares) {
  vector<Rect> rects;
	Scalar colors[6] = {
		{70, 25, 130, 0}, // red
		{90, 90, 180, 0}, // orange
		{100, 185, 185, 0}, // yellow
		{65, 95, 35, 0}, // green
		{140, 25, 0, 0}, // blue
		{200, 175, 160, 0} // white

	};

  char colorchars[6] = {
    'R', 'O', 'Y',
    'G', 'B', 'W'

  };

  for (size_t i = 0; i < squares.size(); i ++) {

    Rect r = boundingRect(Mat(squares[i]));
    if (r.height > 30 && r.width > 30 ) rects.push_back(r);
    Mat roi = image(r);
    Scalar color = mean(roi);
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
    // cout << "rect: " << r << ", delta: " << min  << ", "<< c  << ": " << color << endl;

  }
  // cout << "size: " << squares.size() / 2 << endl;
  bool (*compareFn)(Rect, Rect) = compareRects;
  stable_sort(rects.begin(), rects.end(), compareFn);

  for (size_t k = 0; k < rects.size(); k++) {
    Rect current = rects[k];
    // if (k % 2 == 0) rects.erase(rects.begin() + k);
    /*else*/ cout << k + 1 << " : " << current << endl;

  }
  // cout << "size: " << rects.size() << endl;


}

// compares the coordinates of a rectangle
static bool compareRects(Rect r1, Rect r2) {
  int delta = 25; // allowed difference between the 2 numbers

  if (r1.y + delta > r2.y || r1.y - delta < r2.y) return (r1.x + delta <= r2.x && r1.x - delta >= r2.x);
  return (r1.y + delta < r2.y && r1.y - delta > r2.y);

}

void output_squares(vector<vector<Point> >& squares)
{

    for(size_t j=0; j < squares.size() ; j++ )
    {
      for(size_t i=0; i<4 ; i++) {
           Point* p1 = &squares[j][i];
           cout<<" X= "<<p1->x<<" Y= "<<p1->y<<" ";
       }
       cout<<endl;
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

Scalar Rubik_Face( Mat &image, vector<vector<Point>> &squares )
{
   Point centroid={0,0};
   centroid.x=0;centroid.y=0;
   int N= centers.size();
   for(size_t i=0; i<N; i++)
   {
     centroid.x += centers[i].x;
     centroid.y += centers[i].y;

   }
   centroid.x = centroid.x/N;
   centroid.y = centroid.y/N; 
   //return centroid;
   cout<<" Centroid = "<<centroid.x<<" "<<centroid.y<<endl;
   size_t square_no;
   for( size_t i = 0; i < N; i++)
   {
      if( ( abs(centroid.x - centers[i].x) < 15) && ( abs(centroid.y - centers[i].y) < 15) )
      { 
        square_no = i; break;
      }
   }
   
    // square_no = 5;
 
        Rect r = boundingRect(Mat(squares[square_no]));
        Mat roi = image(r);
        Scalar color = mean(roi); 
        
    /* for(size_t i=0; i < 4; i++)
     {
        cout<<color[i]<<" ";
     } cout<<endl;
     */
    // circle(image, centroid, 25, color, 1, LINE_8, 0);
     
   cout<<"color block identified\n";
   return color;
}

static char detectBlockColor(Scalar color) {
  vector<Rect> rects;
	Scalar colors[6] = {
		{70, 25, 130, 0}, // red
		{90, 90, 180, 0}, // orange
		{100, 185, 185, 0}, // yellow
		{65, 95, 35, 0}, // green
		{140, 25, 0, 0}, // blue
		{200, 175, 160, 0} // white

	};

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
  
/*
    // cout << "rect: " << r << ", delta: " << min  << ", "<< c  << ": " << color << endl;
  // cout << "size: " << squares.size() / 2 << endl;
  bool (*compareFn)(Rect, Rect) = compareRects;
  stable_sort(rects.begin(), rects.end(), compareFn);

  for (size_t k = 0; k < rects.size(); k++) {
    Rect current = rects[k];
    // if (k % 2 == 0) rects.erase(rects.begin() + k);
    /*else*/ //cout << k + 1 << " : " << current << endl;

  //}
  // cout << "size: " << rects.size() << endl;
 

}

static char FaceData[6][9];
enum FaceColor { R, O, Y, G, B, W};


void ExtractColors(char c, Mat &image, vector<vector<Point>> &squares)
{
  cout<<"Recording "<<c<<" face"<<endl;

  //FaceData[c][4] = c; //recording the center face as it has been already detected
        
  for(size_t i=0; i<9; i++)
  {
    
     Rect r = boundingRect(Mat(squares[i]));
     Mat roi = image(r);
     Scalar color = mean(roi);
     FaceData[c][i] = detectBlockColor( color);
  }
  for(size_t i=0; i<9 ; i++){
    cout<<" "<<FaceData[c][i]<<" ";  }
  cout<<endl;

}

//void FaceRecord(char c,  Mat &image, vector<vector<Point>> &squares)
int FaceRecord(Mat &image, vector<vector<Point>> &squares)
{
  static char Faces[6] = { 0,0,0,0,0,0 };   // { R, O, Y, G, B, W} 
  static int x=0;
  char rubik[] = { 'R', 'O', 'Y', 'G', 'B', 'W'};
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
     //c = detectBlockColor( color);
     FaceData[x][i] = detectBlockColor( color);
   }
   cout<<"Face color "<<face_c<<" is recorded, change the face\n";
   x++; //moving to next color
  }
  return x;
  /* 
   if( c == 'R')
   {
     if(!Faces[0])
     {
       //need to record the all the faces
       ExtractColors(c, image, squares);
       Faces[0] = 1;
     }
     else
      cout<<"This face has been recorded\n";
   }
   else if( c == 'O')
     Faces[1] = 1;
   else if( c == 'Y')
     Faces[2] = 1;
   else if( c == 'G')
     Faces[3] = 1;
   else if( c == 'B')
     Faces[4] = 1;
   else if( c == 'W')
     Faces[5] = 1;
   
   */


}


int main() {
   /* vector<vector<Point>> squares;

    static const char* names[] = { "rubik_sq1.png",
        0, 0, 0, 0 };

    for( int i = 0; names[i] != 0; i++ )
    {

       string filename = samples::findFile(names[i]);
        Mat frame = imread(filename, IMREAD_COLOR);
        if( frame.empty() )
        {
            cout << "Couldn't load " << filename << endl;
            continue;
        }
       findSquares(frame, squares);
       output_squares(squares);
       int N = squares.size();
       cout<<"N = "<<N<<endl;
      if(blocks_filter(squares)) {
       cout<<"new frame ******\n";
       cout<<"after filter\n"<<endl;
       output_squares(squares);
       drawSquares(frame, squares);
       imshow("Rubic Detection Demo", frame);
      }
       

      //with centers
      for( size_t k=0;k < centers.size(); k++)
      {
        circle(frame, centers[k], 3, (255, 0, 0), 1, LINE_8, 0);

      } 

       drawSquares(frame, squares);
       imshow("Rubic all squares", frame);
       // imwrite("test.png", frame);
        waitKey(0);
    }*/
   
    Mat frame;
    vector<vector<Point>> squares;
    VideoCapture cap(0);
    static int face_record_count=0; 
    char c;
    //while ((int) squares.size() < 18) {
    while (1) {
      centers.clear();
      cap >> frame;
      if (frame.empty()) return -1;

      findSquares(frame, squares);
      // output_squares(squares);
      int N = squares.size();
     // cout<<"N = "<<N<<endl;
      if(blocks_filter(squares)) {
      
      //checking the centers pattern
      //circle(frame, centers[6], 50, Scalar(255, 0 ,0) , 5, LINE_8, 0);

     // cout<<"new frame ******\n";
      //cout<<"after filter\n"<<endl;
      // output_squares(squares);
      // circle(frame, Rubik_Face(), 3, (255, 0, 0), 1, LINE_8, 0);
     // c = detectBlockColor(Rubik_Face( frame, squares));
      face_record_count = FaceRecord( frame, squares);

      drawSquares(frame, squares);
      if(face_record_count == 6)
      {  goto Faces_collected; }
      }
      imshow("Rubic Detection Demo", frame);
       drawSquares(frame, squares);
      imshow("Rubic all squares", frame);
     // imwrite("test.png", frame);
      waitKey(1);
       
    }
    Faces_collected:
    cout<<"Data collected:"<<endl;


    for(size_t i=0; i<6; i++)
    {
      for(size_t j=0; j<9; j++)
      {  cout<<" "<<FaceData[i][j]<<" "; }
      cout<<endl;
    }
    //detectColors(frame, squares);


/*
    frame = imread("rubik_sq1.png");
     if (frame.empty()) return -1;

      findSquares(frame, squares);
      output_squares(squares);
      cout<<"new frame ******\n";
      drawSquares(frame, squares);
 output_squares(squares);
      imshow("Rubic Detection Demo", frame);
     // imwrite("test.png", frame);
      waitKey();
  */
    return 0;

}
