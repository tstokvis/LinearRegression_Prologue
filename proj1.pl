% predict([X_hat]. [[Dataset]], Y_hat).
% X_hat * linear_regress([[Dataset]]) = Y_hat.
% Linear Regression: B = inv(trans(X)*X) * trans(X) * y
% B*x = y

% -----------------------------------
%
%	VARIABLES
%
% -----------------------------------


% -----------------------------------
%
% 	HELPERS
% 
% -----------------------------------

% ncols(Matrix, ANS) --> Ans returns the number of columns in the given matrix.
ncols([], ANS) :- ANS is 0.
ncols([H|_], ANS) :- ncols_helper(H, 0, ANS1), ANS is ANS1.
ncols_helper([_|[]], ACC, ANS) :- ANS is ACC+1.
ncols_helper([_|T], ACC, ANS) :- NEXT_ACC is ACC+1, ncols_helper(T,NEXT_ACC,ANS).

% scalar_mmul(S, A, ANS) --> times a scalar by a matrix
scalar_mmul(S, [H|T], ANS) :- row_sc_mmul(S,H,ROW), scalar_mmul(S,T,REST), ANS = [ROW|REST].
scalar_mmul(S, [H|[]], ANS) :- row_sc_mmul(S,H,ROW), ANS = [ROW].
row_sc_mmul(S, [H|T], ANS) :- VAL is S*H, row_sc_mmul(S, T, ANS1), ANS = [VAL|ANS1].
row_sc_mmul(S, [H|[]], ANS) :- VAL is S*H, ANS = [VAL].

% remove_column_and_row(ROW, COL, X, ANS) --> removes both the row ROW and column COL from matrix X.
remove_col_and_row(ROW, COL, X, ANS) :- remove_column(COL, X, ANS1), remove_row(ROW, ANS1, ANS2), ANS = ANS2.

% remove_column(N, X, ANS) --> removes column N from matrix X
remove_column(N, [H|[]], ANS) :- remove_column_from_row(N,H,1,ANS1), ANS = [ANS1].
remove_column(N, [H|T], ANS) :- remove_column_from_row(N,H,1,ANS1), remove_column(N,T,ANS2), ANS = [ANS1|ANS2].
remove_column_from_row(N, [_|T], Current, ANS) :- \+dif(N,Current), ANS = T.
remove_column_from_row(N, [H|T], Current, ANS) :- dif(N,Current), Next is Current+1, remove_column_from_row(N,T,Next,ANS1), ANS = [H|ANS1].

% remove_row(N, X, ANS) --> removes row N from matrix X
remove_row(N, X, ANS) :- remove_row_helper(N, 1, X, ANS1), ANS = ANS1.
remove_row_helper(N, N, [_|[]], ANS) :- ANS = [].
remove_row_helper(N, N, [_|T], ANS) :- \+atomic(T), ANS = T.
remove_row_helper(N, CURR, [H|T], ANS) :- dif(CURR, N), NEXT is CURR+1, remove_row_helper(N, NEXT, T, ANS1), ANS = [H|ANS1]. 


% ---------------------------------------------
% Inverse of a Matrix
% inv(X, ANS):
%	- X is a lists of list of numbers, representing a matrix
%	- ANS is the 
%
% NOTE: If Matrix is singular then an "Arithmetic: evaluation error: 'zero_divisor'"  error will be thrown
% ----------------------------------------------

inv([H|[]], ANS) :- ANS = [H].
inv([[H1|[T1]],[H2|[T2]]], ANS) :- atomic(H1), atomic(T1), atomic(H2), atomic(T2), NT1 is (-1*T1), NH2 is (-1*H2), X = [[T2, NT1],[NH2, H1]], det([[H1,T1],[H2,T2]], DET), scalar_mmul(1/DET, X, ANS1), ANS = ANS1.
inv(X, ANS) :- ncols(X, NCOLS), NCOLS>2, mirror_matrix(X, M_ANS), ANS = M_ANS.

% Step 1: Mirror the Matrix.
mirror_matrix(X, ANS) :- mirror_matrix_helper(X, X, 1, 1, ANS).
mirror_matrix_helper([H|T], X, ROW_NUM, 1, ANS) :- mirror_row(H, ROW_NUM, 1, 1, X, ROW_ANS), NEW_ROW is ROW_NUM+1, mirror_matrix_helper(T, X, NEW_ROW, 0, REST), ANS = [ROW_ANS|REST].
mirror_matrix_helper([H|T], X, ROW_NUM, 0, ANS) :- mirror_row(H, ROW_NUM, 1, 0, X, ROW_ANS), NEW_ROW is ROW_NUM+1, mirror_matrix_helper(T, X, NEW_ROW, 1, REST), ANS = [ROW_ANS|REST].
mirror_matrix_helper([H|[]], X, ROW_NUM, 1, ANS) :- mirror_row(H, ROW_NUM, 1, 1, X, ROW_ANS), ANS = [ROW_ANS].
mirror_matrix_helper([H|[]], X, ROW_NUM, 0, ANS) :- mirror_row(H, ROW_NUM, 1, 0, X, ROW_ANS), ANS = [ROW_ANS].

mirror_row([_|T], ROW_NUM, COL_NUM, 1, FullMatrix, ANS) :- remove_col_and_row(ROW_NUM, COL_NUM, FullMatrix, TrimmedMatrix), det(TrimmedMatrix, DET), NEXT_COL is COL_NUM+1, mirror_row(T, ROW_NUM, NEXT_COL, 0, FullMatrix, ANS1), ANS = [DET|ANS1].
mirror_row([_|T], ROW_NUM, COL_NUM, 0, FullMatrix, ANS) :- remove_col_and_row(ROW_NUM, COL_NUM, FullMatrix, TrimmedMatrix), det(TrimmedMatrix, DET), NEXT_COL is COL_NUM+1, mirror_row(T, ROW_NUM, NEXT_COL, 1, FullMatrix, ANS1), NEG_DET is -1*DET, ANS = [NEG_DET|ANS1].
mirror_row([_|[]], ROW_NUM, COL_NUM, 1, FullMatrix, ANS) :- remove_col_and_row(ROW_NUM, COL_NUM, FullMatrix, TrimmedMatrix), det(TrimmedMatrix, DET), ANS = [DET].
mirror_row([_|[]], ROW_NUM, COL_NUM, 0, FullMatrix, ANS) :- remove_col_and_row(ROW_NUM, COL_NUM, FullMatrix, TrimmedMatrix), det(TrimmedMatrix, DET), NEG_DET is -1*DET, ANS = [NEG_DET].

% Step 2: Adjugate.



% ---------------------------------------------
% Determinate of a Matrix 
% det(X, ANS):
% 	- X is a list of list of numbers, representing a matrix 
%	- ANS is the scalar determinate of the matrix
%
% NOTE: If Matrix is singular then an "Arithmetic: evaluation error: 'zero_divisor'"  error will be thrown
% ---------------------------------------------

det([[H|[]]], ANS) :- ANS is H.
det([[H1|[T1]],[H2|[T2]]], ANS) :- atomic(H1), atomic(T1), atomic(H2), atomic(T2), ANS is H1*T2 - T1*H2.
det(X, ANS) :- ncols(X,NCOLS), NCOLS>2, det_large(X, 1, 1, ANS).

det_large([[H|T]|FT], 1, ACC, ANS) :- ACC2 is ACC+1, det_large([T|FT], 0, ACC2, ANS2), remove_column(ACC, FT, SUB), det(SUB,ANS1), ANS is (H*ANS1)+ANS2.
det_large([[H|T]|FT], 0, ACC, ANS) :- ACC2 is ACC+1, det_large([T|FT], 1, ACC2, ANS2), remove_column(ACC, FT, SUB), det(SUB,ANS1), ANS is (-1*H*ANS1)+ANS2.
det_large([[H|[]]|FT], _, ACC, ANS) :- remove_column(ACC, FT, SUB), det(SUB,ANS1), ANS is H*ANS1.


% ---------------------------------------------
% Matrix multiplication (and helpers)
% mmul(A,B,ANS): A*B = ANS
% 	- A&B are a list of lists of numbers, representing a matrix (each inner list is a row).
%	- ANS - the answer to the multiplication.
% 
% NOTE: If the dimensions of the matricies do not agree, the matricies will automatically be trimmed to 
%	the largest dimensions possible that agree (outer most columns or rows will be trimmed).
% ---------------------------------------------

mmul([XH|XT],Y,A) :- row_dev(XH,Y,R), mmul(XT,Y,A2), A = [R|A2].
mmul([XH|XT],Y,A) :- row_dev(XH,Y,R), \+mmul(XT,Y,_), A = [R]. 

row_dev(R, M, A) :- point_dev(R, M, H), remove_head(M, M2), row_dev(R, M2, T), A = [H|T]. 
row_dev(R, M, A) :- point_dev(R, M, H), remove_head(M, M2), \+row_dev(R, M2, _), A = [H].

point_dev([H|T],[[HM|_]|TM], A) :- ANS1 is H*HM, point_dev(T,TM,ANS2), A is ANS1 + ANS2.
point_dev([],[],A) :- A is 0.

remove_head([[_|T]|T2], A) :- remove_head(T2,A2), A = [T|A2].
remove_head([[_|T]|[]], A) :- A = [T].