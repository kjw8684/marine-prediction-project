package com.example.marine.controller;

import com.example.marine.service.MarinePredictorService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import java.util.*;

@RestController
@RequestMapping("/api")
public class LocationInfoController {
    @Autowired
    private MarinePredictorService predictorService;

    @GetMapping("/location-info")
    public Map<String, Object> getLocationInfo(@RequestParam double lat, @RequestParam double lon, @RequestParam String date) {
        // TODO: CMEMS API로 temp, currentSpeed, currentDir 받아오기
        double temp = 20.0; // 샘플값
        double currentSpeed = 1.0;
        double currentDir = 90.0;
        long dateEpoch = java.time.LocalDate.parse(date).toEpochDay();
        List<String> speciesList = Arrays.asList("Engraulis japonicus", "Todarodes pacificus");
        Map<String, Double> probMap = new HashMap<>();
        for (String species : speciesList) {
            double prob = predictorService.predict(lat, lon, dateEpoch, temp, currentSpeed, currentDir, species);
            probMap.put(species, prob);
        }
        Map<String, Object> result = new HashMap<>();
        result.put("lat", lat);
        result.put("lon", lon);
        result.put("date", date);
        result.put("species_probabilities", probMap);
        return result;
    }
}
