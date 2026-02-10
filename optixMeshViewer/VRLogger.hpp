#pragma once

#include <fstream>
#include <string>
#include <iostream>

struct EyePose {
    float position[3];
    float rotation[4]; // x, y, z, w
};

class VRLogger {
public:
    VRLogger(const std::string& filename) : m_filename(filename), m_firstEntry(true) {
        m_file.open(filename);
        if (m_file.is_open()) {
            m_file << "[\n";
        } else {
            std::cerr << "Failed to open log file: " << filename << std::endl;
        }
    }

    ~VRLogger() {
        if (m_file.is_open()) {
            m_file << "\n]";
            m_file.close();
            std::cout << "VR Log saved to " << m_filename << std::endl;
        }
    }

    void log(long long timestamp, const EyePose& left, const EyePose& right) {
        if (!m_file.is_open()) return;

        if (!m_firstEntry) {
            m_file << ",\n";
        }
        m_firstEntry = false;

        m_file << "  {\n";
        m_file << "    \"timestamp\": " << timestamp << ",\n";
        
        m_file << "    \"left_eye\": {\n";
        m_file << "      \"position\": [" << left.position[0] << ", " << left.position[1] << ", " << left.position[2] << "],\n";
        m_file << "      \"rotation\": [" << left.rotation[0] << ", " << left.rotation[1] << ", " << left.rotation[2] << ", " << left.rotation[3] << "]\n";
        m_file << "    },\n";

        m_file << "    \"right_eye\": {\n";
        m_file << "      \"position\": [" << right.position[0] << ", " << right.position[1] << ", " << right.position[2] << "],\n";
        m_file << "      \"rotation\": [" << right.rotation[0] << ", " << right.rotation[1] << ", " << right.rotation[2] << ", " << right.rotation[3] << "]\n";
        m_file << "    }\n";
        m_file << "  }";
        m_file.flush(); // Força a escrita no disco imediatamente
    }

private:
    std::string m_filename;
    std::ofstream m_file;
    bool m_firstEntry;
};