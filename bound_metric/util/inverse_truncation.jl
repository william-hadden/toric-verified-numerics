function prepare_inverse_coeffs_for_subdivision(inverse_coeffs, pdeg::Integer)
    return (;
        A11 = truncate_coeffs_with_tail(inverse_coeffs.A11, pdeg),
        A12 = truncate_coeffs_with_tail(inverse_coeffs.A12, pdeg),
        A22 = truncate_coeffs_with_tail(inverse_coeffs.A22, pdeg),
        D = truncate_coeffs_with_tail(inverse_coeffs.D, pdeg),
    )
end
