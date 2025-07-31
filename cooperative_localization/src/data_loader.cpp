#include "data_loader.h"

// Implementazione del comparatore per ordinare gli eventi
bool EventComparator::operator()(const Event& a, const Event& b) const {
    auto get_timestamp = [](const Event& e) {
        return std::visit([](const auto& arg) { return arg.timestamp; }, e);
    };
    return get_timestamp(a) < get_timestamp(b);
}

bool DataLoader::loadBarcodes(const std::string& filename) {
    std::ifstream file(filename);
    if(!file.is_open()) {
        std::cerr << "Errore: Impossibile aprire il file dei barcode: " << filename << std::endl;
        return false;
    }

    std::string line;
    while(std::getline(file, line) && line.find('#') != std::string::npos);

    do {
        if(line.empty()) continue;
        std::stringstream ss(line);
        int subject, barcode;
        ss >> subject >> barcode;
        barcode_to_subject_[barcode] = subject;
    } while(std::getline(file, line));

    return true;
}

bool DataLoader::loadLandmarkGroundtruth(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Errore: Impossibile aprire il file dei landmark: " << filename << std::endl;
        return false;
    }

    std::string line;
    // Salta le righe di commento (che iniziano con '#')
    while (std::getline(file, line) && line.find('#') != std::string::npos);

    // Leggi i dati reali
    do {
        if (line.empty()) continue;
        std::stringstream ss(line);
        Landmark lm;
        double std_x, std_y;
        ss >> lm.id >> lm.pos.x() >> lm.pos.y() >> std_x >> std_y;
        landmark_map[lm.id] = lm;
    } while (std::getline(file, line));
    
    std::cout << "Caricati " << landmark_map.size() << " landmark." << std::endl;
    return true;
}

bool DataLoader::loadOdometry(const std::string& filename, int robot_id) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Errore: Impossibile aprire il file di odometria: " << filename << std::endl;
        return false;
    }
    std::string line;
    while (std::getline(file, line) && line.find('#') != std::string::npos);
    
    do {
        if (line.empty()) continue;
        std::stringstream ss(line);
        OdometryData odom;
        odom.robot_id = robot_id;
        ss >> odom.timestamp >> odom.v >> odom.w;
        event_queue.emplace_back(odom);
    } while (std::getline(file, line));

    return true;
}

bool DataLoader::loadMeasurements(const std::string& filename, int robot_id) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Errore: Impossibile aprire il file delle misure: " << filename << std::endl;
        return false;
    }
    std::string line;
    while (std::getline(file, line) && line.find('#') != std::string::npos);
    
    do {
        if (line.empty()) continue;
        std::stringstream ss(line);
        MeasurementData meas;
        meas.observer_id = robot_id;
        int barcode;
        ss >> meas.timestamp >> barcode >> meas.range >> meas.bearing;
        auto it = barcode_to_subject_.find(barcode);
        meas.subject_id = (it != barcode_to_subject_.end()) ? it->second : barcode;
        event_queue.emplace_back(meas);
    } while (std::getline(file, line));

    return true;
}

const std::vector<Event>& DataLoader::getEventQueue() const {
    // Ordina la coda prima di restituirla se non è già stato fatto
    // (potrebbe essere più efficiente farlo una sola volta dopo aver caricato tutto)
    std::sort(const_cast<DataLoader*>(this)->event_queue.begin(), const_cast<DataLoader*>(this)->event_queue.end(), EventComparator());
    return event_queue;
}

const std::map<int, Landmark>& DataLoader::getLandmarks() const {
    return landmark_map;
}