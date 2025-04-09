class LocationQueries:
    @staticmethod
    def get_city_coverage():
        return """
        WITH doctor_cities AS (
            SELECT DISTINCT city 
            FROM doctors 
            WHERE city IS NOT NULL
        ),
        hospital_locations AS (
            SELECT DISTINCT location 
            FROM hospitals 
            WHERE location IS NOT NULL
        )
        SELECT 
            c.city,
            COUNT(DISTINCT d.did) as doctor_count,
            COUNT(DISTINCT h.hid) as hospital_count,
            COUNT(DISTINCT cl.cid) as clinic_count
        FROM doctor_cities c
        LEFT JOIN doctors d ON d.city = c.city
        LEFT JOIN hospitals h ON h.location = c.city
        LEFT JOIN clinics cl ON cl.location = c.city
        GROUP BY c.city
        ORDER BY doctor_count DESC;
        """