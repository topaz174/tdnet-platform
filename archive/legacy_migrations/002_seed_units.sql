INSERT INTO units (id, currency, scale, unit_code, note)
VALUES 
    (1, 'JPY', 6, 'JPY_Mil', 'Japanese Yen, in millions'),
    (2, 'JPY', 0, 'JPY', 'Japanese Yen, unscaled'),
    (3, 'JPY', 3, 'JPY_Thou', 'Japanese Yen, in thousands'),
    (4, 'USD', 6, 'USD_Mil', 'US Dollars, in millions'),
    (5, 'SHR', 0, 'Shares', 'Number of shares'),
    (6, 'PUR', 0, 'Pure', 'Pure number/ratio without unit')
ON CONFLICT (unit_code) DO NOTHING; 