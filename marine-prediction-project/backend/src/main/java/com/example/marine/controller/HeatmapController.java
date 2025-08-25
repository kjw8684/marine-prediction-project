package com.example.marine.controller;

import com.example.marine.service.MarinePredictorService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import java.util.*;
import java.io.*;
import java.nio.file.*;
import java.util.stream.*;
import org.apache.commons.csv.*;

@RestController
@RequestMapping("/api")
public class HeatmapController {
    @Autowired
    private MarinePredictorService predictorService;

    @GetMapping("/heatmap")
    public List<Map<String, Object>> getHeatmap(@RequestParam String species, @RequestParam String date) {
        List<Map<String, Object>> result = new ArrayList<>();
        // 환경 데이터 캐시 파일 경로
        String envCachePath = "ai/data/env_cache.csv";
        try (Reader reader = Files.newBufferedReader(Paths.get(envCachePath))) {
            Iterable<CSVRecord> records = CSVFormat.DEFAULT.withFirstRecordAsHeader().parse(reader);
            for (CSVRecord record : records) {
                double lat = Double.parseDouble(record.get("lat"));
                double lon = Double.parseDouble(record.get("lon"));
                String envDate = record.get("date");
                double temp = Double.parseDouble(record.get("temp"));
                double currentSpeed = Double.parseDouble(record.get("current_speed"));
                double currentDir = Double.parseDouble(record.get("current_dir"));
                // date 파라미터와 envDate가 같은 경우만 사용 (혹은 최신 데이터만 사용)
                // 여기서는 최신 env_cache.csv만 사용
                long dateEpoch = java.time.LocalDate.parse(date).toEpochDay();
                double prob = predictorService.predict(lat, lon, dateEpoch, temp, currentSpeed, currentDir, species);
                Map<String, Object> point = new HashMap<>();
                point.put("lat", lat);
                point.put("lon", lon);
                point.put("probability", prob);
                result.add(point);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return result;
    }
}
