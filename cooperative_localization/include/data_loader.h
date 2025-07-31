#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <vector>
#include <variant>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <Eigen/Dense>

// Struttura per un dato di odometria
struct OdometryData {
    double timestamp;
    double v; // velocità lineare
    double w; // velocità angolare
    int robot_id;
};

// Struttura per una misura (landmark o altro robot)
struct MeasurementData {
    double timestamp;
    int observer_id;   // ID del robot che osserva
    int subject_id;    // ID dell'oggetto/robot osservato
    double range;
    double bearing;
};

// Struttura per la posizione reale (per la valutazione)
struct Groundtruth {
    double timestamp;
    Eigen::Vector3d pose; // (x, y, theta)
};

// Struttura per la posizione di un landmark
struct Landmark {
    int id;
    Eigen::Vector2d pos; // (x, y)
};

// Usiamo std::variant per creare un "evento" generico
// Questo ci permette di mettere tutti i tipi di dati in un'unica coda
using Event = std::variant<OdometryData, MeasurementData>;

// Funzione di confronto per ordinare gli eventi per timestamp
struct EventComparator {
    bool operator()(const Event& a, const Event& b) const;
};

class DataLoader {
public:
    // Carica la mappa dei landmark
    bool loadLandmarkGroundtruth(const std::string& filename);

    // Carica i dati di odometria per un dato robot
    bool loadOdometry(const std::string& filename, int robot_id);

    // Carica i dati di misurazione per un dato robot
    bool loadMeasurements(const std::string& filename, int robot_id);

    // Restituisce la coda di eventi ordinata
    const std::vector<Event>& getEventQueue() const;
    
    // Restituisce la mappa dei landmark
    const std::map<int, Landmark>& getLandmarks() const;

private:
    std::vector<Event> event_queue;
    std::map<int, Landmark> landmark_map;
};

#endif // DATA_LOADER_H