#include "geometry.h"
#include <cmath>
#include <iostream>


Line::Line(Point origin, double dx, double dy){
    O = origin;
    double mag = std::sqrt(dx*dx + dy*dy);
    dir.x = dx/mag;
    dir.y = dy/mag;
}

Point Line::project_point_on_line(const Point& point_to_project) const{
    //this method function returns a point which is the result of projecting the parameter onto the line
    // vector from O to p
    double vx = point_to_project.x - O.x;
    double vy = point_to_project.y - O.y;

    // projection scalar = dot(v, dir)
    double t = vx*dir.x + vy*dir.y;

    // projected point
    return Point( O.x + dir.x * t, O.y + dir.y * t);
}

double Line::projection_error(const Point& p1, const Point& p2) const{
    //calculates the difference between the euclidean distance of two points when they are projected onto the line
    //and between the distance of the original points

    Point q1 = this->project_point_on_line(p1);
    Point q2 = this->project_point_on_line(p2);

    double original_distance = euclidean_distance(p1, p2);
    double projected_distance = euclidean_distance(q1, q2);
    //return std::fabs(original_distance - projected_distance);
    return std::fabs(original_distance - projected_distance); //trying out a square loss function to see if the rotation works there
}



double euclidean_distance(const Point& p1, const Point& p2){
    return std::hypot(p1.x - p2.x, p1.y - p2.y);
}