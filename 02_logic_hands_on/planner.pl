% ---------------------------------------------------------------------------
% planner.pl
% Prolog verifier for the warehouse planning domain (optional extension).
% ---------------------------------------------------------------------------

% --- Task 6: warehouse connectivity facts ---
connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).

can_move(X,Y) :-
    connected(X,Y).

% --- Task 7: independent check of a proposed plan's moves ---
valid_move(X,Y) :-
    connected(X,Y).

% --- Task 8: simple chained-rule inference example ---
wet_road.

slippery :-
    wet_road.

reduce_speed :-
    slippery.
