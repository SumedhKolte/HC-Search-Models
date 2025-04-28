class MedicalQueries:
    @staticmethod
    def get_disease_symptom_mapping():
        return """
        SELECT 
            d.name as disease_name,
            array_agg(s.name) as symptoms,
            d.common_names,
            d.tags as disease_tags
        FROM diseases d
        JOIN LATERAL unnest(d.related_diseases) disease_id ON TRUE
        JOIN symptoms s ON s.symptom_id = disease_id
        GROUP BY d.disease_id, d.name, d.common_names, d.tags;
        """

    @staticmethod
    def get_doctor_specialization_stats():
        return """
        SELECT 
            specialization,
            COUNT(*) as doctor_count,
            AVG(NULLIF(rating::numeric, 0)) as avg_rating,
            AVG(NULLIF(consultantfee::numeric, 0)) as avg_consultantfee
        FROM doctors
        WHERE specialization IS NOT NULL
        GROUP BY specialization
        ORDER BY doctor_count DESC;
        """