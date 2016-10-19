# ArbitraryNorm

Arbitrary colorscale normalization for matplotlib

Update: This is now being included in a matplotlib with a [pull request](https://github.com/matplotlib/matplotlib/pull/7294) under review.

"ArbitraryNorm" is a class built on the class "Normalize" that provides a much easier way to create arbitrary non linear normalization, for both the positive and negative directions independently. This is achieved by passing a pair of non linear functions (and their inverse) as an argument in the constructor. This two functions will be used to independently normalise the positive and negative data, into the [0,1] range. It also allows fixing the value of the colormap that will be used for zero, letting the user decided which fraction of the colorbar ([0,1] range) they want to use for the positive and negative range, effectively letting the used fixing the color assigned to the zero value, even for non symmetric intervals.
In includes similar versions specialized in the positive-only and negative-only directions.

There are also classes built inheriting from the "ArbitraryNorm" classes, implementing a root normalization, by indicating just the degree of the root for the positive and the negative range.

Overall these classes, as they are much more powerful and allow much more control over the intervals, make the existing class "PowerNorm" obsolete.

Examples of use are also included.

Copyright © 2015 Alvaro Sanchez Gonzalez
