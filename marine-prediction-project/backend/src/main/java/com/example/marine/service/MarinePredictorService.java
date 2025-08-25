package com.example.marine.service;

import org.jpmml.evaluator.*;
import org.jpmml.model.PMMLUtil;
import org.springframework.stereotype.Service;
import java.io.FileInputStream;
import java.util.*;

@Service
public class MarinePredictorService {
    private Evaluator evaluator;
    private Map<String, Integer> speciesLabelMap;

    public MarinePredictorService() throws Exception {
        PMML pmml = PMMLUtil.unmarshal(new FileInputStream("ai/model/marine_model.pmml"));
        this.evaluator = ModelEvaluatorFactory.newInstance().newModelEvaluator(pmml);
        // TODO: species 라벨 인코딩 맵을 Python에서 저장한 값과 일치시켜야 함
        this.speciesLabelMap = new HashMap<>();
        speciesLabelMap.put("Engraulis japonicus", 0);
        speciesLabelMap.put("Todarodes pacificus", 1);
    }

    public double predict(double lat, double lon, long date, double temp, double currentSpeed, double currentDir, String species) {
        Integer speciesId = speciesLabelMap.get(species);
        if (speciesId == null) return 0.0;
        Map<FieldName, Object> arguments = new LinkedHashMap<>();
        arguments.put(new FieldName("lat"), lat);
        arguments.put(new FieldName("lon"), lon);
        arguments.put(new FieldName("date"), date);
        arguments.put(new FieldName("temp"), temp);
        arguments.put(new FieldName("current_speed"), currentSpeed);
        arguments.put(new FieldName("current_dir"), currentDir);
        arguments.put(new FieldName("species"), speciesId);
        Map<FieldName, ?> result = evaluator.evaluate(arguments);
        Object prob = result.get(evaluator.getTargetField());
        if (prob instanceof Double) return (Double) prob;
        if (prob instanceof Number) return ((Number) prob).doubleValue();
        return 0.0;
    }
}
