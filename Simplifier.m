
function [simple_expression] = Simplifier(expr, variablen)
syms(variablen);
expression = str2sym(expr);
a =simplify(expression);
simple_expression = latex(a);
end
